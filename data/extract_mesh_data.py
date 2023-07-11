import os
import json
import xml.etree.ElementTree as ET


def main():
    in_file_path = os.path.join('data', 'mesh_data_2023.xml')
    out_file_path = os.path.join('data', 'mesh_crosswalk.json')
    crosswalk = extract_mesh_data_from_xml(in_file_path)
    with open(out_file_path, 'w') as f:
        json.dump(crosswalk, f, indent=4, sort_keys=True)


def extract_mesh_data_from_xml(file_path):
    """ Extract crosswalk from mesh descriptor unique id to tree number(s)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = {}
    
    for record in root.findall('.//DescriptorRecord'):
        descriptor_ui = record.find('DescriptorUI').text
        tree_numbers = [
            tn.text for tn in record.findall('.//TreeNumberList/TreeNumber')
        ]
        if descriptor_ui not in data:
            data[descriptor_ui] = tree_numbers

    return data


if __name__ == '__main__':
    main()
    