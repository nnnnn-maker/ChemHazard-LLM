"""
危化品泄漏事故研判 - 扩展数据爬取脚本
爬取毒性数据（LD50/LC50）和物理化学性质（沸点、闪点、蒸气压等）
用于泄漏物质状态预测和毒性等级预测
"""

import asyncio
import aiohttp
import json
import os
import csv
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Set


class ExtendedPropertyScraper:
    def __init__(
        self,
        output_dir: str = "pubchem_data",
        max_concurrent: int = 3,
        retry_times: int = 3
    ):
        """
        初始化扩展属性爬虫

        Args:
            output_dir: 输出目录
            max_concurrent: 最大并发数
            retry_times: 重试次数
        """
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.retry_times = retry_times

        os.makedirs(output_dir, exist_ok=True)

        self.completed_cids: Set[int] = set()
        self.failed_cids: List[int] = []
        self.all_data: List[Dict] = []

        self._load_progress()

    def _load_progress(self):
        """加载进度"""
        progress_file = f"{self.output_dir}/extended_progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.completed_cids = set(progress.get("completed_cids", []))
                self.failed_cids = progress.get("failed_cids", [])
            print(f"已加载进度: {len(self.completed_cids)} 个化合物已完成")

        data_file = f"{self.output_dir}/extended_properties.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                self.all_data = json.load(f)
            print(f"已加载 {len(self.all_data)} 条数据")

    def _save_progress(self):
        """保存进度"""
        progress_file = f"{self.output_dir}/extended_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "completed_cids": list(self.completed_cids),
                "failed_cids": self.failed_cids,
                "last_update": datetime.now().isoformat()
            }, f, indent=2)

    def _parse_physical_properties(self, json_data: Dict, cid: int) -> Dict:
        """
        解析物理化学性质数据

        提取：沸点、熔点、闪点、蒸气压、密度、水溶性等
        """
        result = {
            "CID": cid,
            "BoilingPoint": None,      # 沸点 (°C)
            "MeltingPoint": None,      # 熔点 (°C)
            "FlashPoint": None,        # 闪点 (°C)
            "VaporPressure": None,     # 蒸气压 (mmHg)
            "Density": None,           # 密度 (g/cm³)
            "WaterSolubility": None,   # 水溶性 (mg/L)
            "LogP": None,              # 辛醇-水分配系数
            "PhysicalState": None,     # 物理状态 (solid/liquid/gas)
            "Has_Physical_Data": False
        }

        try:
            record = json_data.get("Record", {})
            sections = record.get("Section", [])

            for section in sections:
                if section.get("TOCHeading") == "Chemical and Physical Properties":
                    for subsec in section.get("Section", []):
                        heading = subsec.get("TOCHeading", "")

                        if heading == "Experimental Properties":
                            self._extract_experimental_props(subsec, result)

                        elif heading == "Computed Properties":
                            self._extract_computed_props(subsec, result)

            if any([result["BoilingPoint"], result["MeltingPoint"],
                    result["FlashPoint"], result["VaporPressure"]]):
                result["Has_Physical_Data"] = True

        except Exception as e:
            print(f"CID {cid} 物理性质解析错误: {e}")

        return result

    def _extract_experimental_props(self, section: Dict, result: Dict):
        """提取实验测定的物理性质"""
        for subsec in section.get("Section", []):
            heading = subsec.get("TOCHeading", "")

            for info in subsec.get("Information", []):
                value = info.get("Value", {})
                string_val = ""

                if "StringWithMarkup" in value:
                    for item in value["StringWithMarkup"]:
                        string_val = item.get("String", "")
                        break
                elif "Number" in value:
                    string_val = str(value["Number"][0])

                if not string_val:
                    continue

                # 提取数值
                num_match = re.search(r'[-+]?\d*\.?\d+', string_val)
                num_val = float(num_match.group()) if num_match else None

                if "Boiling Point" in heading and result["BoilingPoint"] is None:
                    result["BoilingPoint"] = num_val

                elif "Melting Point" in heading and result["MeltingPoint"] is None:
                    result["MeltingPoint"] = num_val

                elif "Flash Point" in heading and result["FlashPoint"] is None:
                    result["FlashPoint"] = num_val

                elif "Vapor Pressure" in heading and result["VaporPressure"] is None:
                    result["VaporPressure"] = num_val

                elif "Density" in heading and result["Density"] is None:
                    result["Density"] = num_val

                elif "Solubility" in heading and result["WaterSolubility"] is None:
                    result["WaterSolubility"] = num_val

                elif "Physical Description" in heading:
                    desc_lower = string_val.lower()
                    if "liquid" in desc_lower:
                        result["PhysicalState"] = "liquid"
                    elif "gas" in desc_lower or "vapor" in desc_lower:
                        result["PhysicalState"] = "gas"
                    elif "solid" in desc_lower or "powder" in desc_lower or "crystal" in desc_lower:
                        result["PhysicalState"] = "solid"

    def _extract_computed_props(self, section: Dict, result: Dict):
        """提取计算的物理性质"""
        for info in section.get("Information", []):
            name = info.get("Name", "")
            value = info.get("Value", {})

            if "Number" in value:
                num_val = value["Number"][0]

                if "LogP" in name and result["LogP"] is None:
                    result["LogP"] = num_val

    def _parse_toxicity_data(self, json_data: Dict, cid: int) -> Dict:
        """
        解析毒性数据

        提取：LD50、LC50、毒性等级等
        """
        result = {
            "CID": cid,
            "LD50_oral": None,         # 经口LD50 (mg/kg)
            "LD50_dermal": None,       # 经皮LD50 (mg/kg)
            "LC50_inhalation": None,   # 吸入LC50 (ppm 或 mg/L)
            "Toxicity_Class": None,    # 毒性等级 (1-6, GHS分类)
            "IDLH": None,              # 立即危害生命健康浓度
            "Has_Toxicity_Data": False
        }

        try:
            record = json_data.get("Record", {})
            sections = record.get("Section", [])

            for section in sections:
                if section.get("TOCHeading") == "Toxicity":
                    result["Has_Toxicity_Data"] = True
                    self._extract_toxicity_values(section, result)

                elif section.get("TOCHeading") == "Safety and Hazards":
                    for subsec in section.get("Section", []):
                        if subsec.get("TOCHeading") == "Toxicity":
                            result["Has_Toxicity_Data"] = True
                            self._extract_toxicity_values(subsec, result)

        except Exception as e:
            print(f"CID {cid} 毒性数据解析错误: {e}")

        # 根据LD50计算毒性等级
        if result["LD50_oral"]:
            result["Toxicity_Class"] = self._calculate_toxicity_class(result["LD50_oral"])

        return result

    def _extract_toxicity_values(self, section: Dict, result: Dict):
        """提取毒性数值"""
        for subsec in section.get("Section", []):
            heading = subsec.get("TOCHeading", "")

            for info in subsec.get("Information", []):
                value = info.get("Value", {})
                string_val = ""

                if "StringWithMarkup" in value:
                    for item in value["StringWithMarkup"]:
                        string_val = item.get("String", "")
                        break

                if not string_val:
                    continue

                string_lower = string_val.lower()

                # 提取 LD50/LC50 数值
                num_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg/kg|mg/l|ppm)', string_lower)
                if num_match:
                    num_val = float(num_match.group(1))

                    if "ld50" in string_lower:
                        if "oral" in string_lower and result["LD50_oral"] is None:
                            result["LD50_oral"] = num_val
                        elif "dermal" in string_lower and result["LD50_dermal"] is None:
                            result["LD50_dermal"] = num_val

                    elif "lc50" in string_lower:
                        if result["LC50_inhalation"] is None:
                            result["LC50_inhalation"] = num_val

                    elif "idlh" in string_lower:
                        if result["IDLH"] is None:
                            result["IDLH"] = num_val

    def _calculate_toxicity_class(self, ld50_oral: float) -> int:
        """
        根据经口LD50计算GHS毒性等级

        GHS急性毒性分类（经口）:
        1类: LD50 ≤ 5 mg/kg
        2类: 5 < LD50 ≤ 50 mg/kg
        3类: 50 < LD50 ≤ 300 mg/kg
        4类: 300 < LD50 ≤ 2000 mg/kg
        5类: 2000 < LD50 ≤ 5000 mg/kg
        6类: LD50 > 5000 mg/kg (低毒)
        """
        if ld50_oral <= 5:
            return 1
        elif ld50_oral <= 50:
            return 2
        elif ld50_oral <= 300:
            return 3
        elif ld50_oral <= 2000:
            return 4
        elif ld50_oral <= 5000:
            return 5
        else:
            return 6

    async def _fetch_single(
        self,
        session: aiohttp.ClientSession,
        cid: int,
        semaphore: asyncio.Semaphore
    ) -> Optional[Dict]:
        """获取单个化合物的扩展数据"""
        if cid in self.completed_cids:
            return None

        # 获取完整的化合物数据
        url = f"{self.base_url}/data/compound/{cid}/JSON"

        async with semaphore:
            for attempt in range(self.retry_times):
                try:
                    await asyncio.sleep(0.5)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            data = await response.json()

                            # 解析物理性质
                            physical = self._parse_physical_properties(data, cid)

                            # 解析毒性数据
                            toxicity = self._parse_toxicity_data(data, cid)

                            # 合并结果
                            result = {**physical}
                            for key, val in toxicity.items():
                                if key != "CID":
                                    result[key] = val

                            self.completed_cids.add(cid)
                            return result

                        elif response.status == 404:
                            result = {
                                "CID": cid,
                                "Has_Physical_Data": False,
                                "Has_Toxicity_Data": False
                            }
                            self.completed_cids.add(cid)
                            return result

                        elif response.status == 503:
                            wait_time = (attempt + 1) * 5
                            print(f"CID {cid}: 服务器繁忙，等待 {wait_time} 秒")
                            await asyncio.sleep(wait_time)

                except asyncio.TimeoutError:
                    print(f"CID {cid}: 超时，重试 {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"CID {cid}: 错误 {e}")
                    await asyncio.sleep(2)

            self.failed_cids.append(cid)
            return None

    async def scrape(self, cid_list: List[int]) -> List[Dict]:
        """主爬取函数"""
        pending_cids = [cid for cid in cid_list if cid not in self.completed_cids]
        total = len(pending_cids)

        print(f"开始爬取扩展属性数据")
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
            batch_size = 50
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

                # 显示进度
                completed = min(i + batch_size, total)
                progress = completed / total * 100
                phys_count = sum(1 for d in self.all_data if d.get("Has_Physical_Data"))
                tox_count = sum(1 for d in self.all_data if d.get("Has_Toxicity_Data"))
                print(f"进度: {completed}/{total} ({progress:.1f}%) - "
                      f"物理数据: {phys_count}, 毒性数据: {tox_count}")

                self._save_progress()
                self._save_data()

        return self.all_data

    def _save_data(self):
        """保存数据"""
        filepath = f"{self.output_dir}/extended_properties.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

    def save_results(self):
        """保存最终结果"""
        if not self.all_data:
            print("没有数据可保存")
            return

        self._save_data()
        print(f"已保存 {len(self.all_data)} 条数据到 extended_properties.json")

        # 保存 CSV
        filepath = f"{self.output_dir}/extended_properties.csv"
        fieldnames = [
            'CID', 'BoilingPoint', 'MeltingPoint', 'FlashPoint', 'VaporPressure',
            'Density', 'WaterSolubility', 'LogP', 'PhysicalState',
            'LD50_oral', 'LD50_dermal', 'LC50_inhalation', 'Toxicity_Class', 'IDLH',
            'Has_Physical_Data', 'Has_Toxicity_Data'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.all_data)

        print(f"已保存 CSV 到 {filepath}")

        # 统计
        phys_count = sum(1 for d in self.all_data if d.get("Has_Physical_Data"))
        tox_count = sum(1 for d in self.all_data if d.get("Has_Toxicity_Data"))
        print(f"\n数据统计:")
        print(f"  有物理性质数据: {phys_count}/{len(self.all_data)} ({phys_count/len(self.all_data)*100:.1f}%)")
        print(f"  有毒性数据: {tox_count}/{len(self.all_data)} ({tox_count/len(self.all_data)*100:.1f}%)")

    def merge_all_data(self, compounds_file: str, ghs_file: str, output_file: str = "full_dataset.csv"):
        """
        合并所有数据：基础属性 + GHS数据 + 扩展属性

        Args:
            compounds_file: 基础化合物数据
            ghs_file: GHS数据
            output_file: 输出文件名
        """
        print("\n合并所有数据...")

        # 读取基础数据
        compounds = {}
        with open(compounds_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                compounds[int(row['CID'])] = row

        # 读取 GHS 数据
        ghs_data = {}
        with open(ghs_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ghs_data[int(row['CID'])] = row

        # 创建扩展数据索引
        extended_index = {item['CID']: item for item in self.all_data}

        # 合并
        merged = []
        for cid, compound in compounds.items():
            row = compound.copy()

            # 添加 GHS 数据
            ghs = ghs_data.get(cid, {})
            for key in ['Has_GHS_Data', 'Signal_Word', 'H_Statements']:
                row[key] = ghs.get(key, '')

            # 添加扩展数据
            ext = extended_index.get(cid, {})
            for key in ['BoilingPoint', 'MeltingPoint', 'FlashPoint', 'VaporPressure',
                        'Density', 'WaterSolubility', 'PhysicalState',
                        'LD50_oral', 'LD50_dermal', 'LC50_inhalation', 'Toxicity_Class', 'IDLH']:
                row[key] = ext.get(key, '')

            merged.append(row)

        # 保存
        output_path = f"{self.output_dir}/{output_file}"
        fieldnames = list(merged[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)

        print(f"已保存完整数据集到 {output_path}")
        print(f"总计 {len(merged)} 条记录")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='危化品扩展属性爬取工具')
    parser.add_argument('--input', type=str, default='pubchem_data/compounds.csv',
                        help='输入的化合物数据文件')
    parser.add_argument('--output', type=str, default='pubchem_data',
                        help='输出目录')
    parser.add_argument('--concurrent', type=int, default=3,
                        help='最大并发数')
    parser.add_argument('--merge', action='store_true',
                        help='爬取完成后合并所有数据')

    args = parser.parse_args()

    # 读取 CID 列表
    cid_list = []
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid_list.append(int(row['CID']))

    print(f"从 {args.input} 读取了 {len(cid_list)} 个 CID")

    # 创建爬虫
    scraper = ExtendedPropertyScraper(
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
        scraper.merge_all_data(
            args.input,
            f"{args.output}/ghs_data.csv"
        )

    print(f"\n爬取完成! 总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
