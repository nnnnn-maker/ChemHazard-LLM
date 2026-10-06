"""
Extended property scraper for hazardous-chemical assessment.
Retrieves toxicity data (LD50/LC50) and physicochemical properties
(such as boiling point, flash point, and vapor pressure).
Originally developed for substance-state and toxicity-class prediction.
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
        Initialize the extended property scraper.

        Args:
            output_dir: Output directory.
            max_concurrent: Maximum number of concurrent requests.
            retry_times: Number of retry attempts.
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
        """Load previous progress."""
        progress_file = f"{self.output_dir}/extended_progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                self.completed_cids = set(progress.get("completed_cids", []))
                self.failed_cids = progress.get("failed_cids", [])
            print(f"Progress loaded: {len(self.completed_cids)} compounds completed")

        data_file = f"{self.output_dir}/extended_properties.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                self.all_data = json.load(f)
            print(f"Loaded {len(self.all_data)} records")

    def _save_progress(self):
        """Save current progress."""
        progress_file = f"{self.output_dir}/extended_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "completed_cids": list(self.completed_cids),
                "failed_cids": self.failed_cids,
                "last_update": datetime.now().isoformat()
            }, f, indent=2)

    def _parse_physical_properties(self, json_data: Dict, cid: int) -> Dict:
        """
        Parse physicochemical property data.

        Extract boiling point, melting point, flash point, vapor pressure,
        density, water solubility, and related properties.
        """
        result = {
            "CID": cid,
            "BoilingPoint": None,      # Boiling point (°C)
            "MeltingPoint": None,      # Melting point (°C)
            "FlashPoint": None,        # Flash point (°C)
            "VaporPressure": None,     # Vapor pressure (mmHg)
            "Density": None,           # Density (g/cm³)
            "WaterSolubility": None,   # Water solubility (mg/L)
            "LogP": None,              # Octanol-water partition coefficient
            "PhysicalState": None,     # Physical state (solid/liquid/gas)
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
            print(f"CID {cid} physical-property parse error: {e}")

        return result

    def _extract_experimental_props(self, section: Dict, result: Dict):
        """Extract experimentally measured physical properties."""
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

                # Extract a numeric value.
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
        """Extract computed physical properties."""
        for info in section.get("Information", []):
            name = info.get("Name", "")
            value = info.get("Value", {})

            if "Number" in value:
                num_val = value["Number"][0]

                if "LogP" in name and result["LogP"] is None:
                    result["LogP"] = num_val

    def _parse_toxicity_data(self, json_data: Dict, cid: int) -> Dict:
        """
        Parse toxicity data.

        Extract LD50, LC50, toxicity class, and related properties.
        """
        result = {
            "CID": cid,
            "LD50_oral": None,         # Oral LD50 (mg/kg)
            "LD50_dermal": None,       # Dermal LD50 (mg/kg)
            "LC50_inhalation": None,   # Inhalation LC50 (ppm or mg/L)
            "Toxicity_Class": None,    # Toxicity class (1-6, GHS classification)
            "IDLH": None,              # Immediately dangerous to life or health
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
            print(f"CID {cid} toxicity-data parse error: {e}")

        # Calculate toxicity class from oral LD50.
        if result["LD50_oral"]:
            result["Toxicity_Class"] = self._calculate_toxicity_class(result["LD50_oral"])

        return result

    def _extract_toxicity_values(self, section: Dict, result: Dict):
        """Extract numeric toxicity values."""
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

                # Extract LD50/LC50 values.
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
        Calculate the GHS toxicity class from oral LD50.

        Oral acute toxicity classes used by this script:
        Class 1: LD50 <= 5 mg/kg
        Class 2: 5 < LD50 <= 50 mg/kg
        Class 3: 50 < LD50 <= 300 mg/kg
        Class 4: 300 < LD50 <= 2000 mg/kg
        Class 5: 2000 < LD50 <= 5000 mg/kg
        Class 6: LD50 > 5000 mg/kg (lower toxicity)
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
        """Fetch extended properties for one compound."""
        if cid in self.completed_cids:
            return None

        # Fetch the full compound record.
        url = f"{self.base_url}/data/compound/{cid}/JSON"

        async with semaphore:
            for attempt in range(self.retry_times):
                try:
                    await asyncio.sleep(0.5)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Parse physical properties.
                            physical = self._parse_physical_properties(data, cid)

                            # Parse toxicity data.
                            toxicity = self._parse_toxicity_data(data, cid)

                            # Merge the results.
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
                            print(f"CID {cid}: server busy; waiting {wait_time} seconds")
                            await asyncio.sleep(wait_time)

                except asyncio.TimeoutError:
                    print(f"CID {cid}: timeout; retry {attempt + 1}/{self.retry_times}")
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"CID {cid}: error {e}")
                    await asyncio.sleep(2)

            self.failed_cids.append(cid)
            return None

    async def scrape(self, cid_list: List[int]) -> List[Dict]:
        """Scrape extended properties for the requested CIDs."""
        pending_cids = [cid for cid in cid_list if cid not in self.completed_cids]
        total = len(pending_cids)

        print("Starting extended-property scrape")
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

                # Display progress.
                completed = min(i + batch_size, total)
                progress = completed / total * 100
                phys_count = sum(1 for d in self.all_data if d.get("Has_Physical_Data"))
                tox_count = sum(1 for d in self.all_data if d.get("Has_Toxicity_Data"))
                print(f"Progress: {completed}/{total} ({progress:.1f}%) - "
                      f"physical data: {phys_count}, toxicity data: {tox_count}")

                self._save_progress()
                self._save_data()

        return self.all_data

    def _save_data(self):
        """Save data."""
        filepath = f"{self.output_dir}/extended_properties.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

    def save_results(self):
        """Save the final results."""
        if not self.all_data:
            print("No data to save")
            return

        self._save_data()
        print(f"Saved {len(self.all_data)} records to extended_properties.json")

        # Save CSV.
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

        print(f"Saved CSV to {filepath}")

        # Display statistics.
        phys_count = sum(1 for d in self.all_data if d.get("Has_Physical_Data"))
        tox_count = sum(1 for d in self.all_data if d.get("Has_Toxicity_Data"))
        print("\nDataset statistics:")
        print(f"  Records with physical data: {phys_count}/{len(self.all_data)} ({phys_count/len(self.all_data)*100:.1f}%)")
        print(f"  Records with toxicity data: {tox_count}/{len(self.all_data)} ({tox_count/len(self.all_data)*100:.1f}%)")

    def merge_all_data(self, compounds_file: str, ghs_file: str, output_file: str = "full_dataset.csv"):
        """
        Merge basic compound, GHS, and extended-property data.

        Args:
            compounds_file: Basic compound data file.
            ghs_file: GHS data file.
            output_file: Output file name.
        """
        print("\nMerging all data...")

        # Read basic compound data.
        compounds = {}
        with open(compounds_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                compounds[int(row['CID'])] = row

        # Read GHS data.
        ghs_data = {}
        with open(ghs_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ghs_data[int(row['CID'])] = row

        # Index extended-property data by CID.
        extended_index = {item['CID']: item for item in self.all_data}

        # Merge the data.
        merged = []
        for cid, compound in compounds.items():
            row = compound.copy()

            # Add GHS data.
            ghs = ghs_data.get(cid, {})
            for key in ['Has_GHS_Data', 'Signal_Word', 'H_Statements']:
                row[key] = ghs.get(key, '')

            # Add extended-property data.
            ext = extended_index.get(cid, {})
            for key in ['BoilingPoint', 'MeltingPoint', 'FlashPoint', 'VaporPressure',
                        'Density', 'WaterSolubility', 'PhysicalState',
                        'LD50_oral', 'LD50_dermal', 'LC50_inhalation', 'Toxicity_Class', 'IDLH']:
                row[key] = ext.get(key, '')

            merged.append(row)

        # Save the merged data.
        output_path = f"{self.output_dir}/{output_file}"
        fieldnames = list(merged[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)

        print(f"Complete dataset saved to {output_path}")
        print(f"Total records: {len(merged)}")


async def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Hazardous-chemical extended-property scraper')
    parser.add_argument('--input', type=str, default='pubchem_data/compounds.csv',
                        help='Input compound data file')
    parser.add_argument('--output', type=str, default='pubchem_data',
                        help='Output directory')
    parser.add_argument('--concurrent', type=int, default=3,
                        help='Maximum number of concurrent requests')
    parser.add_argument('--merge', action='store_true',
                        help='Merge all data after scraping')

    args = parser.parse_args()

    # Read the CID list.
    cid_list = []
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid_list.append(int(row['CID']))

    print(f"Read {len(cid_list)} CIDs from {args.input}")

    # Create the scraper.
    scraper = ExtendedPropertyScraper(
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
        scraper.merge_all_data(
            args.input,
            f"{args.output}/ghs_data.csv"
        )

    print(f"\nScrape complete! Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
