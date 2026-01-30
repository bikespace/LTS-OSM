#!/usr/bin/env python3
"""
Usage:
python build_query.py region key value
     
Example usage: 
python build_query.py eastyork wikidata Q167585
"""

import argparse
from pathlib import Path
from string import Template

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build OSM Overpass query file based on location')
    parser.add_argument('region', type=str, nargs='?',
                        help='Name of region (used for file saving)')
    parser.add_argument('key', type=str, default="wikidata", nargs='?', 
                        help='OSM key for relation to download')
    parser.add_argument('value', type=str, nargs='?',
                        help='OSM value for relation to download')
    
    args = parser.parse_args()
    
    region = args.region
    key = args.key
    value = args.value
    
    # build query file from template
    template_path = Path(__file__).with_name("query_template.overpass")
    template_text = template_path.read_text()
    query = Template(template_text).substitute(key=key, value=value)

    output_path = Path(f"{region}.query")
    output_path.write_text(query)
