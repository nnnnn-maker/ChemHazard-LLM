"""
PubChem GHS hazard data scraper.
Adds GHS classifications, hazard statements, and precautionary statements to compound data.
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
        Initialize the GHS data scraper.

        Args:
            output_dir: Output directory.
            max_concurrent: Maximum concurrent requests (use a low value for GHS data).
            retry_times: Number of retry attempts.
            delay: Delay between requests, in seconds.
        """
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.retry_times = retry_times
        self.delay = delay

        os.makedirs(output_dir, exist_ok=True)

        # Track progress.
        self.completed_cids: Set[int] = set()
        self.failed_cids: List[int] = []
        self.all_data: List[Dict] = []

        self._load_progress()

    def _load_progress(self):
        """Load previous progress."""
        progress_file = f"{self.output_dir}/ghs_progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.completed_cids = set(progress.get("completed_cids", []))
                self.failed_cids = progress.get("failed_cids", [])
            print(f"Progress loaded: {len(self.completed_cids)} compounds completed")

        # Load previously saved data.
        data_file = f"{self.output_dir}/ghs_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                self.all_data = json.load(f)
            print(f"Loaded {len(self.all_data)} saved GHS records")

    def _save_progress(self):
        """Save current progress."""
        progress_file = f"{self.output_dir}/ghs_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "completed_cids": list(self.completed_cids),
                "failed_cids": self.failed_cids,
                "last_update": datetime.now().isoformat()
            }, f, indent=2)

    def _parse_ghs_data(self, json_data: Dict, cid: int) -> Optional[Dict]:
        """
        Parse GHS data returned by PubChem.

        Args:
            json_data: JSON response from the PubChem API.
            cid: Compound ID.

        Returns:
            Parsed GHS data dictionary.
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
            print(f"CID {cid} parse error: {e}")

        return result

    def _extract_ghs_info(self, ghs_section: Dict, result: Dict):
        """Extract details from the GHS classification section."""
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

        # Extract GHS classifications.
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
        Fetch GHS data for one compound.

        Args:
            session: aiohttp session.
            cid: Compound ID.
            semaphore: Semaphore controlling concurrency.

        Returns:
            GHS data dictionary.
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
                            # No GHS data was returned for this compound.
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
                            print(f"CID {cid}: server busy; waiting {wait_time} seconds")
                            await asyncio.sleep(wait_time)

                        else:
                            print(f"CID {cid}: HTTP {response.status}")

                except asyncio.TimeoutError:
                    print(f"CID {cid}: timeout; retry {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"CID {cid}: error {e}")
                    await asyncio.sleep(2)

            self.failed_cids.append(cid)
            return None

    async def scrape(self, cid_list: List[int]) -> List[Dict]:
        """
        Scrape GHS data for the requested CIDs.

        Args:
            cid_list: CIDs to scrape.

        Returns:
            All retrieved GHS records.
        """
        # Exclude completed CIDs.
        pending_cids = [cid for cid in cid_list if cid not in self.completed_cids]
        total = len(pending_cids)

        print("Starting GHS data scrape")
        print(f"Total compounds: {len(cid_list)}")
        print(f"Pending: {total}")
        print(f"Completed: {len(self.completed_cids)}")
        print("-" * 50)

        if total == 0:
            print("All compounds have been processed")
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
                        print(f"Task error: {result}")

                # Display progress.
                completed = min(i + batch_size, total)
                progress = completed / total * 100
                ghs_count = sum(1 for d in self.all_data if d.get("Has_GHS_Data"))
                print(f"Progress: {completed}/{total} ({progress:.1f}%) - "
                      f"records with GHS data: {ghs_count}/{len(self.all_data)}")

                # Save progress and data.
                self._save_progress()
                self._save_data()

        return self.all_data

    def _save_data(self):
        """Save data to a file."""
        filepath = f"{self.output_dir}/ghs_data.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

    def save_results(self):
        """Save the final results."""
        if not self.all_data:
            print("No data to save")
            return

        # Save JSON.
        self._save_data()
        print(f"Saved {len(self.all_data)} GHS records to ghs_data.json")

        # Save CSV.
        self._save_csv()

        # Display statistics.
        ghs_count = sum(1 for d in self.all_data if d.get("Has_GHS_Data"))
        print(f"Compounds with GHS data: {ghs_count}/{len(self.all_data)} "
              f"({ghs_count/len(self.all_data)*100:.1f}%)")

    def _save_csv(self):
        """Save data in CSV format."""
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

        print(f"Saved CSV to {filepath}")

    def merge_with_compounds(self, compounds_file: str, output_file: str = "compounds_with_ghs.csv"):
        """
        Merge GHS data with the original compound data.

        Args:
            compounds_file: Path to the original compound data file.
            output_file: Output file name.
        """
        # Read the original data.
        compounds = []
        with open(compounds_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            compounds = list(reader)

        # Index GHS data by CID.
        ghs_index = {item['CID']: item for item in self.all_data}

        # Merge the data.
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

        # Save the merged data.
        output_path = f"{self.output_dir}/{output_file}"
        fieldnames = list(merged[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)

        print(f"Merged data saved to {output_path}")
        print(f"Total records: {len(merged)}")


async def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='PubChem GHS hazard data scraper')
    parser.add_argument('--input', type=str, default='pubchem_data/compounds.csv',
                        help='Input compound data file')
    parser.add_argument('--output', type=str, default='pubchem_data',
                        help='Output directory')
    parser.add_argument('--concurrent', type=int, default=3,
                        help='Maximum concurrent requests (default: 3)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge data after scraping')

    args = parser.parse_args()

    # Read the CID list.
    cid_list = []
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid_list.append(int(row['CID']))

    print(f"Read {len(cid_list)} CIDs from {args.input}")

    # Create the scraper.
    scraper = GHSScraper(
        output_dir=args.output,
        max_concurrent=args.concurrent
    )

    # Start the scrape.
    start_time = time.time()
    await scraper.scrape(cid_list)
    elapsed = time.time() - start_time

    # Save results.
    scraper.save_results()

    # Merge data.
    if args.merge:
        scraper.merge_with_compounds(args.input)

    print(f"\nScrape complete! Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
