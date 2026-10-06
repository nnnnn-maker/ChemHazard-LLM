"""
PubChem GHS 危险性数据爬取脚本
为现有化合物数据补充 GHS 分类、H声明、P声明等危险性信息
"""

import asyncio
import aiohttp
import json
import os
import time
import csv
from datetime import datetime
from typing import List, Dict, Optional, Set
import xml.etree.ElementTree as ET


class GHSScraper:
    def __init__(
        self,
        output_dir: str = "pubchem_data",
        max_concurrent: int = 3,
        retry_times: int = 3,
        delay: float = 0.5
    ):
        """
        初始化 GHS 数据爬虫

        Args:
            output_dir: 输出目录
            max_concurrent: 最大并发请求数（GHS数据较大，建议较低并发）
            retry_times: 失败重试次数
            delay: 请求间隔（秒）
        """
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.retry_times = retry_times
        self.delay = delay

        os.makedirs(output_dir, exist_ok=True)

        # 进度追踪
        self.completed_cids: Set[int] = set()
        self.failed_cids: List[int] = []
        self.all_data: List[Dict] = []

        self._load_progress()

    def _load_progress(self):
        """加载之前的进度"""
        progress_file = f"{self.output_dir}/ghs_progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.completed_cids = set(progress.get("completed_cids", []))
                self.failed_cids = progress.get("failed_cids", [])
            print(f"已加载进度: {len(self.completed_cids)} 个化合物已完成")

        # 加载已保存的数据
        data_file = f"{self.output_dir}/ghs_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                self.all_data = json.load(f)
            print(f"已加载 {len(self.all_data)} 条已保存的GHS数据")

    def _save_progress(self):
        """保存当前进度"""
        progress_file = f"{self.output_dir}/ghs_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "completed_cids": list(self.completed_cids),
                "failed_cids": self.failed_cids,
                "last_update": datetime.now().isoformat()
            }, f, indent=2)

    def _parse_ghs_data(self, json_data: Dict, cid: int) -> Optional[Dict]:
        """
        解析 PubChem 返回的 GHS 数据

        Args:
            json_data: PubChem API 返回的 JSON 数据
            cid: 化合物 ID

        Returns:
            解析后的 GHS 数据字典
        """
        result = {
            "CID": cid,
            "GHS_Classifications": [],
            "H_Statements": [],
            "P_Statements": [],
            "Signal_Word": None,
            "Pictograms": [],
            "Has_GHS_Data": False
        }

        try:
            record = json_data.get("Record", {})
            sections = record.get("Section", [])

            for section in sections:
                if section.get("TOCHeading") == "Safety and Hazards":
                    for subsection in section.get("Section", []):
                        if subsection.get("TOCHeading") == "Hazards Identification":
                            for sub2 in subsection.get("Section", []):
                                if sub2.get("TOCHeading") == "GHS Classification":
                                    result["Has_GHS_Data"] = True
                                    self._extract_ghs_info(sub2, result)

        except Exception as e:
            print(f"CID {cid} 解析错误: {e}")

        return result

    def _extract_ghs_info(self, ghs_section: Dict, result: Dict):
        """从 GHS 分类部分提取详细信息"""
        for info in ghs_section.get("Information", []):
            name = info.get("Name", "")
            value = info.get("Value", {})

            if "GHS Hazard Statements" in name:
                for item in value.get("StringWithMarkup", []):
                    statement = item.get("String", "")
                    if statement:
                        result["H_Statements"].append(statement)

            elif "Precautionary Statement" in name:
                for item in value.get("StringWithMarkup", []):
                    statement = item.get("String", "")
                    if statement:
                        result["P_Statements"].append(statement)

            elif "Signal" in name:
                for item in value.get("StringWithMarkup", []):
                    result["Signal_Word"] = item.get("String", "")

            elif "Pictogram" in name:
                for item in value.get("StringWithMarkup", []):
                    pic = item.get("String", "")
                    if pic:
                        result["Pictograms"].append(pic)

        # 提取 GHS 分类
        for subsec in ghs_section.get("Section", []):
            heading = subsec.get("TOCHeading", "")
            if heading:
                result["GHS_Classifications"].append(heading)

    async def _fetch_single(
        self,
        session: aiohttp.ClientSession,
        cid: int,
        semaphore: asyncio.Semaphore
    ) -> Optional[Dict]:
        """
        获取单个化合物的 GHS 数据

        Args:
            session: aiohttp 会话
            cid: 化合物 ID
            semaphore: 并发控制信号量

        Returns:
            GHS 数据字典
        """
        if cid in self.completed_cids:
            return None

        url = f"{self.base_url}/data/compound/{cid}/JSON?heading=GHS+Classification"

        async with semaphore:
            for attempt in range(self.retry_times):
                try:
                    await asyncio.sleep(self.delay)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = self._parse_ghs_data(data, cid)
                            self.completed_cids.add(cid)
                            return result

                        elif response.status == 404:
                            # 该化合物没有 GHS 数据
                            result = {
                                "CID": cid,
                                "GHS_Classifications": [],
                                "H_Statements": [],
                                "P_Statements": [],
                                "Signal_Word": None,
                                "Pictograms": [],
                                "Has_GHS_Data": False
                            }
                            self.completed_cids.add(cid)
                            return result

                        elif response.status == 503:
                            wait_time = (attempt + 1) * 5
                            print(f"CID {cid}: 服务器繁忙，等待 {wait_time} 秒")
                            await asyncio.sleep(wait_time)

                        else:
                            print(f"CID {cid}: HTTP {response.status}")

                except asyncio.TimeoutError:
                    print(f"CID {cid}: 超时，重试 {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"CID {cid}: 错误 {e}")
                    await asyncio.sleep(2)

            self.failed_cids.append(cid)
            return None

    async def scrape(self, cid_list: List[int]) -> List[Dict]:
        """
        主爬取函数

        Args:
            cid_list: 要爬取的 CID 列表

        Returns:
            所有 GHS 数据
        """
        # 过滤已完成的
        pending_cids = [cid for cid in cid_list if cid not in self.completed_cids]
        total = len(pending_cids)

        print(f"开始爬取 GHS 数据")
        print(f"总计: {len(cid_list)} 个化合物")
        print(f"待爬取: {total} 个")
        print(f"已完成: {len(self.completed_cids)} 个")
        print("-" * 50)

        if total == 0:
            print("所有化合物已完成爬取")
            return self.all_data

        semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)

        async with aiohttp.ClientSession(connector=connector) as session:
            batch_size = 100
            for i in range(0, total, batch_size):
                batch = pending_cids[i:i + batch_size]

                tasks = [
                    self._fetch_single(session, cid, semaphore)
                    for cid in batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict):
                        self.all_data.append(result)
                    elif isinstance(result, Exception):
                        print(f"任务异常: {result}")

                # 显示进度
                completed = min(i + batch_size, total)
                progress = completed / total * 100
                ghs_count = sum(1 for d in self.all_data if d.get("Has_GHS_Data"))
                print(f"进度: {completed}/{total} ({progress:.1f}%) - "
                      f"有GHS数据: {ghs_count}/{len(self.all_data)}")

                # 保存进度和数据
                self._save_progress()
                self._save_data()

        return self.all_data

    def _save_data(self):
        """保存数据到文件"""
        filepath = f"{self.output_dir}/ghs_data.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

    def save_results(self):
        """保存最终结果"""
        if not self.all_data:
            print("没有数据可保存")
            return

        # 保存 JSON
        self._save_data()
        print(f"已保存 {len(self.all_data)} 条 GHS 数据到 ghs_data.json")

        # 保存 CSV
        self._save_csv()

        # 统计信息
        ghs_count = sum(1 for d in self.all_data if d.get("Has_GHS_Data"))
        print(f"有 GHS 数据的化合物: {ghs_count}/{len(self.all_data)} "
              f"({ghs_count/len(self.all_data)*100:.1f}%)")

    def _save_csv(self):
        """保存为 CSV 格式"""
        filepath = f"{self.output_dir}/ghs_data.csv"

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                'CID', 'Has_GHS_Data', 'Signal_Word',
                'GHS_Classifications', 'H_Statements', 'P_Statements', 'Pictograms'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in self.all_data:
                row = {
                    'CID': item['CID'],
                    'Has_GHS_Data': item['Has_GHS_Data'],
                    'Signal_Word': item['Signal_Word'] or '',
                    'GHS_Classifications': '|'.join(item['GHS_Classifications']),
                    'H_Statements': '|'.join(item['H_Statements']),
                    'P_Statements': '|'.join(item['P_Statements']),
                    'Pictograms': '|'.join(item['Pictograms'])
                }
                writer.writerow(row)

        print(f"已保存 CSV 到 {filepath}")

    def merge_with_compounds(self, compounds_file: str, output_file: str = "compounds_with_ghs.csv"):
        """
        将 GHS 数据与原始化合物数据合并

        Args:
            compounds_file: 原始化合物数据文件路径
            output_file: 输出文件名
        """
        # 读取原始数据
        compounds = []
        with open(compounds_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            compounds = list(reader)

        # 创建 GHS 数据索引
        ghs_index = {item['CID']: item for item in self.all_data}

        # 合并数据
        merged = []
        for compound in compounds:
            cid = int(compound['CID'])
            ghs = ghs_index.get(cid, {})

            merged_row = compound.copy()
            merged_row['Has_GHS_Data'] = ghs.get('Has_GHS_Data', False)
            merged_row['Signal_Word'] = ghs.get('Signal_Word', '')
            merged_row['GHS_Classifications'] = '|'.join(ghs.get('GHS_Classifications', []))
            merged_row['H_Statements'] = '|'.join(ghs.get('H_Statements', []))
            merged_row['P_Statements'] = '|'.join(ghs.get('P_Statements', []))
            merged_row['Pictograms'] = '|'.join(ghs.get('Pictograms', []))
            merged.append(merged_row)

        # 保存合并后的数据
        output_path = f"{self.output_dir}/{output_file}"
        fieldnames = list(merged[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)

        print(f"已合并数据并保存到 {output_path}")
        print(f"总计 {len(merged)} 条记录")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PubChem GHS 危险性数据爬取工具')
    parser.add_argument('--input', type=str, default='pubchem_data/compounds.csv',
                        help='输入的化合物数据文件')
    parser.add_argument('--output', type=str, default='pubchem_data',
                        help='输出目录')
    parser.add_argument('--concurrent', type=int, default=3,
                        help='最大并发数 (默认: 3)')
    parser.add_argument('--merge', action='store_true',
                        help='爬取完成后合并数据')

    args = parser.parse_args()

    # 读取 CID 列表
    cid_list = []
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid_list.append(int(row['CID']))

    print(f"从 {args.input} 读取了 {len(cid_list)} 个 CID")

    # 创建爬虫
    scraper = GHSScraper(
        output_dir=args.output,
        max_concurrent=args.concurrent
    )

    # 开始爬取
    start_time = time.time()
    await scraper.scrape(cid_list)
    elapsed = time.time() - start_time

    # 保存结果
    scraper.save_results()

    # 合并数据
    if args.merge:
        scraper.merge_with_compounds(args.input)

    print(f"\n爬取完成! 总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
