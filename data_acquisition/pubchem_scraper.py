"""
Asynchronous PubChem batch data scraper.
Supports resuming, retries, and progress tracking.
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
        Initialize the scraper.

        Args:
            output_dir: Output directory.
            batch_size: Number of CIDs per request (maximum 100).
            max_concurrent: Maximum number of concurrent requests.
            retry_times: Number of retry attempts.
            delay: Delay between requests, in seconds.
        """
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.output_dir = output_dir
        self.batch_size = min(batch_size, 100)
        self.max_concurrent = max_concurrent
        self.retry_times = retry_times
        self.delay = delay

        # Create the output directory.
        os.makedirs(output_dir, exist_ok=True)

        # Track progress.
        self.completed_batches = set()
        self.failed_cids = []
        self.all_data = []

        # Load previous progress.
        self._load_progress()

    def _load_progress(self):
        """Load previous progress."""
        progress_file = f"{self.output_dir}/progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.completed_batches = set(progress.get("completed_batches", []))
                self.failed_cids = progress.get("failed_cids", [])
            print(f"Progress loaded: {len(self.completed_batches)} batches completed")

    def _save_progress(self):
        """Save current progress."""
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
        Fetch a batch of compound data asynchronously.

        Args:
            session: aiohttp session.
            cids: List of CIDs.
            batch_id: Batch ID.
            semaphore: Semaphore controlling concurrency.

        Returns:
            List of compound records.
        """
        # Skip completed batches.
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
                            # Some CIDs may not exist; skip the batch.
                            print(f"Batch {batch_id}: some CIDs do not exist; skipping")
                            self.failed_cids.extend(cids)
                            return None

                        elif response.status == 503:
                            # Wait before retrying when the server is busy.
                            wait_time = (attempt + 1) * 5
                            print(f"Batch {batch_id}: server busy; retrying in {wait_time} seconds")
                            await asyncio.sleep(wait_time)

                        else:
                            print(f"Batch {batch_id}: HTTP {response.status}")

                except asyncio.TimeoutError:
                    print(f"Batch {batch_id}: timeout; retry {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"Batch {batch_id}: error {e}; retry {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

            # All retry attempts failed.
            self.failed_cids.extend(cids)
            return None

    async def scrape(self, cid_list: List[int]) -> List[Dict]:
        """
        Scrape the requested CIDs.

        Args:
            cid_list: CIDs to scrape.

        Returns:
            All retrieved compound records.
        """
        total_cids = len(cid_list)
        total_batches = (total_cids + self.batch_size - 1) // self.batch_size

        print(f"Starting scrape of {total_cids} compounds in {total_batches} batches")
        print(f"Configuration: batch size={self.batch_size}, maximum concurrency={self.max_concurrent}")
        print("-" * 50)

        # Create batches.
        batches = []
        for i in range(0, total_cids, self.batch_size):
            batch_cids = cid_list[i:i + self.batch_size]
            batch_id = i // self.batch_size
            batches.append((batch_id, batch_cids))

        # Control concurrency.
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create the HTTP session.
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Process batches in groups, saving progress after each group.
            group_size = 50  # Save progress every 50 batches.
            all_results = []

            for group_start in range(0, len(batches), group_size):
                group_batches = batches[group_start:group_start + group_size]

                # Create tasks.
                tasks = [
                    self._fetch_batch(session, cids, batch_id, semaphore)
                    for batch_id, cids in group_batches
                ]

                # Execute tasks.
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results.
                for result in results:
                    if isinstance(result, list):
                        all_results.extend(result)
                    elif isinstance(result, Exception):
                        print(f"Task error: {result}")

                # Display progress.
                completed = group_start + len(group_batches)
                progress = completed / len(batches) * 100
                print(f"Progress: {completed}/{len(batches)} ({progress:.1f}%) - {len(all_results)} records retrieved")

                # Save progress.
                self._save_progress()

                # Save checkpoints periodically.
                if len(all_results) > 0 and completed % 100 == 0:
                    self._save_checkpoint(all_results, completed)

        self.all_data = all_results
        return all_results

    def _save_checkpoint(self, data: List[Dict], batch_num: int):
        """Save checkpoint data."""
        filename = f"{self.output_dir}/checkpoint_{batch_num}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Checkpoint saved: {filename}")

    def save_results(self, filename: str = "compounds.json"):
        """
        Save the final results.

        Args:
            filename: Output file name.
        """
        if not self.all_data:
            print("No data to save")
            return

        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.all_data)} compound records to {filepath}")

        # Save failed CIDs.
        if self.failed_cids:
            failed_file = f"{self.output_dir}/failed_cids.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_cids, f)
            print(f"Failed CIDs saved to {failed_file}")

    def save_as_csv(self, filename: str = "compounds.csv"):
        """
        Save the results in CSV format.

        Args:
            filename: Output file name.
        """
        if not self.all_data:
            print("No data to save")
            return

        import csv

        filepath = f"{self.output_dir}/{filename}"

        # Collect all field names.
        fieldnames = set()
        for item in self.all_data:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_data)

        print(f"Saved {len(self.all_data)} compound records to {filepath}")

    def get_statistics(self) -> Dict:
        """Return scrape statistics."""
        return {
            "total_compounds": len(self.all_data),
            "completed_batches": len(self.completed_batches),
            "failed_cids": len(self.failed_cids)
        }


async def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Asynchronous PubChem batch data scraper')
    parser.add_argument('--start', type=int, default=1, help='Starting CID (default: 1)')
    parser.add_argument('--end', type=int, default=1000, help='Ending CID (default: 1000)')
    parser.add_argument('--file', type=str, help='Read CIDs from a file (one CID per line)')
    parser.add_argument('--output', type=str, default='pubchem_data', help='Output directory (default: pubchem_data)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size (default: 100; maximum: 100)')
    parser.add_argument('--concurrent', type=int, default=5, help='Maximum concurrent requests (default: 5)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both', help='Output format (default: both)')

    args = parser.parse_args()

    # Determine which CIDs to retrieve.
    if args.file:
        # Read CIDs from a file.
        with open(args.file, 'r') as f:
            cid_list = [int(line.strip()) for line in f if line.strip().isdigit()]
        print(f"Read {len(cid_list)} CIDs from {args.file}")
    else:
        # Use a CID range.
        cid_list = list(range(args.start, args.end + 1))
        print(f"CID range: {args.start} - {args.end}")

    # Create the scraper.
    scraper = PubChemAsyncScraper(
        output_dir=args.output,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent
    )

    # Start the scrape.
    start_time = time.time()
    await scraper.scrape(cid_list)
    elapsed = time.time() - start_time

    # Save results.
    if args.format in ['json', 'both']:
        scraper.save_results("compounds.json")
    if args.format in ['csv', 'both']:
        scraper.save_as_csv("compounds.csv")

    # Display statistics.
    stats = scraper.get_statistics()
    print("\n" + "=" * 50)
    print("Scrape complete!")
    print(f"Total compounds: {stats['total_compounds']}")
    print(f"Completed batches: {stats['completed_batches']}")
    print(f"Failed CIDs: {stats['failed_cids']}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average rate: {stats['total_compounds'] / elapsed:.2f} records/second")


if __name__ == "__main__":
    asyncio.run(main())
