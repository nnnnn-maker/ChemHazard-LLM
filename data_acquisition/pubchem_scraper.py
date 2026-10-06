"""
PubChem 异步批量数据爬取脚本
支持断点续传、错误重试、进度保存
"""

import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional


class PubChemAsyncScraper:
    def __init__(
        self,
        output_dir: str = "pubchem_data",
        batch_size: int = 100,
        max_concurrent: int = 5,
        retry_times: int = 3,
        delay: float = 0.3
    ):
        """
        初始化爬虫

        Args:
            output_dir: 输出目录
            batch_size: 每批请求的CID数量（最大100）
            max_concurrent: 最大并发请求数
            retry_times: 失败重试次数
            delay: 请求间隔（秒）
        """
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.output_dir = output_dir
        self.batch_size = min(batch_size, 100)
        self.max_concurrent = max_concurrent
        self.retry_times = retry_times
        self.delay = delay

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 进度追踪
        self.completed_batches = set()
        self.failed_cids = []
        self.all_data = []

        # 加载之前的进度
        self._load_progress()

    def _load_progress(self):
        """加载之前的进度"""
        progress_file = f"{self.output_dir}/progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.completed_batches = set(progress.get("completed_batches", []))
                self.failed_cids = progress.get("failed_cids", [])
            print(f"已加载进度: {len(self.completed_batches)} 个批次已完成")

    def _save_progress(self):
        """保存当前进度"""
        progress_file = f"{self.output_dir}/progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "completed_batches": list(self.completed_batches),
                "failed_cids": self.failed_cids,
                "last_update": datetime.now().isoformat()
            }, f, indent=2)

    async def _fetch_batch(
        self,
        session: aiohttp.ClientSession,
        cids: List[int],
        batch_id: int,
        semaphore: asyncio.Semaphore
    ) -> Optional[List[Dict]]:
        """
        异步获取一批化合物数据

        Args:
            session: aiohttp会话
            cids: CID列表
            batch_id: 批次ID
            semaphore: 并发控制信号量

        Returns:
            化合物数据列表
        """
        # 跳过已完成的批次
        if batch_id in self.completed_batches:
            return None

        cid_str = ",".join(map(str, cids))
        url = (
            f"{self.base_url}/compound/cid/{cid_str}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,"
            "CanonicalSMILES,InChI,InChIKey,XLogP,TPSA,"
            "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,"
            "HeavyAtomCount,Complexity/JSON"
        )

        async with semaphore:
            for attempt in range(self.retry_times):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "PropertyTable" in data:
                                self.completed_batches.add(batch_id)
                                return data["PropertyTable"]["Properties"]

                        elif response.status == 404:
                            # 部分CID不存在，尝试单个获取
                            print(f"批次 {batch_id}: 部分CID不存在，跳过")
                            self.failed_cids.extend(cids)
                            return None

                        elif response.status == 503:
                            # 服务器繁忙，等待后重试
                            wait_time = (attempt + 1) * 5
                            print(f"批次 {batch_id}: 服务器繁忙，等待 {wait_time} 秒后重试")
                            await asyncio.sleep(wait_time)

                        else:
                            print(f"批次 {batch_id}: HTTP {response.status}")

                except asyncio.TimeoutError:
                    print(f"批次 {batch_id}: 超时，重试 {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"批次 {batch_id}: 错误 {e}，重试 {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

            # 所有重试都失败
            self.failed_cids.extend(cids)
            return None

    async def scrape(self, cid_list: List[int]) -> List[Dict]:
        """
        主爬取函数

        Args:
            cid_list: 要爬取的CID列表

        Returns:
            所有化合物数据
        """
        total_cids = len(cid_list)
        total_batches = (total_cids + self.batch_size - 1) // self.batch_size

        print(f"开始爬取 {total_cids} 个化合物，共 {total_batches} 个批次")
        print(f"配置: 批次大小={self.batch_size}, 最大并发={self.max_concurrent}")
        print("-" * 50)

        # 创建批次
        batches = []
        for i in range(0, total_cids, self.batch_size):
            batch_cids = cid_list[i:i + self.batch_size]
            batch_id = i // self.batch_size
            batches.append((batch_id, batch_cids))

        # 并发控制
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # 创建HTTP会话
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 分组处理，每组完成后保存进度
            group_size = 50  # 每50个批次保存一次
            all_results = []

            for group_start in range(0, len(batches), group_size):
                group_batches = batches[group_start:group_start + group_size]

                # 创建任务
                tasks = [
                    self._fetch_batch(session, cids, batch_id, semaphore)
                    for batch_id, cids in group_batches
                ]

                # 执行任务
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理结果
                for result in results:
                    if isinstance(result, list):
                        all_results.extend(result)
                    elif isinstance(result, Exception):
                        print(f"任务异常: {result}")

                # 显示进度
                completed = group_start + len(group_batches)
                progress = completed / len(batches) * 100
                print(f"进度: {completed}/{len(batches)} ({progress:.1f}%) - 已获取 {len(all_results)} 条数据")

                # 保存进度
                self._save_progress()

                # 定期保存数据
                if len(all_results) > 0 and completed % 100 == 0:
                    self._save_checkpoint(all_results, completed)

        self.all_data = all_results
        return all_results

    def _save_checkpoint(self, data: List[Dict], batch_num: int):
        """保存检查点数据"""
        filename = f"{self.output_dir}/checkpoint_{batch_num}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"检查点已保存: {filename}")

    def save_results(self, filename: str = "compounds.json"):
        """
        保存最终结果

        Args:
            filename: 输出文件名
        """
        if not self.all_data:
            print("没有数据可保存")
            return

        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

        print(f"已保存 {len(self.all_data)} 条化合物数据到 {filepath}")

        # 保存失败的CID
        if self.failed_cids:
            failed_file = f"{self.output_dir}/failed_cids.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_cids, f)
            print(f"失败的CID已保存到 {failed_file}")

    def save_as_csv(self, filename: str = "compounds.csv"):
        """
        保存为CSV格式

        Args:
            filename: 输出文件名
        """
        if not self.all_data:
            print("没有数据可保存")
            return

        import csv

        filepath = f"{self.output_dir}/{filename}"

        # 获取所有字段
        fieldnames = set()
        for item in self.all_data:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_data)

        print(f"已保存 {len(self.all_data)} 条化合物数据到 {filepath}")

    def get_statistics(self) -> Dict:
        """获取爬取统计信息"""
        return {
            "total_compounds": len(self.all_data),
            "completed_batches": len(self.completed_batches),
            "failed_cids": len(self.failed_cids)
        }


async def main():
    """主函数 - 使用示例"""
    import argparse

    parser = argparse.ArgumentParser(description='PubChem 异步批量数据爬取工具')
    parser.add_argument('--start', type=int, default=1, help='起始CID (默认: 1)')
    parser.add_argument('--end', type=int, default=1000, help='结束CID (默认: 1000)')
    parser.add_argument('--file', type=str, help='从文件读取CID列表 (每行一个CID)')
    parser.add_argument('--output', type=str, default='pubchem_data', help='输出目录 (默认: pubchem_data)')
    parser.add_argument('--batch-size', type=int, default=100, help='批次大小 (默认: 100, 最大: 100)')
    parser.add_argument('--concurrent', type=int, default=5, help='最大并发数 (默认: 5)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both', help='输出格式 (默认: both)')

    args = parser.parse_args()

    # 确定CID列表
    if args.file:
        # 从文件读取
        with open(args.file, 'r') as f:
            cid_list = [int(line.strip()) for line in f if line.strip().isdigit()]
        print(f"从文件 {args.file} 读取了 {len(cid_list)} 个CID")
    else:
        # 使用范围
        cid_list = list(range(args.start, args.end + 1))
        print(f"CID范围: {args.start} - {args.end}")

    # 创建爬虫实例
    scraper = PubChemAsyncScraper(
        output_dir=args.output,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent
    )

    # 开始爬取
    start_time = time.time()
    await scraper.scrape(cid_list)
    elapsed = time.time() - start_time

    # 保存结果
    if args.format in ['json', 'both']:
        scraper.save_results("compounds.json")
    if args.format in ['csv', 'both']:
        scraper.save_as_csv("compounds.csv")

    # 显示统计
    stats = scraper.get_statistics()
    print("\n" + "=" * 50)
    print("爬取完成!")
    print(f"总化合物数: {stats['total_compounds']}")
    print(f"完成批次数: {stats['completed_batches']}")
    print(f"失败CID数: {stats['failed_cids']}")
    print(f"总耗时: {elapsed:.2f} 秒")
    print(f"平均速度: {stats['total_compounds'] / elapsed:.2f} 条/秒")


if __name__ == "__main__":
    asyncio.run(main())
