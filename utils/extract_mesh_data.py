import os
import json
import xml.etree.ElementTree as ET


# To run this script, you'll need to have the file utils/mesh_data_2023.xml
# (available at the following adress: https://nlmpubs.nlm.nih.gov/projects/mesh/)
# in the same folder as the script. Running this script produces the files
# mesh_crosswalk.json and mesh_crosswalk_inverted.json


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
        descriptor_name = record.find('.//DescriptorName/String').text
        tree_numbers = [
            tn.text for tn in record.findall('.//TreeNumberList/TreeNumber')
        ]
        if descriptor_ui not in data:
            data[descriptor_ui] = tree_numbers
        if descriptor_name not in data:
            data[descriptor_name] = tree_numbers

    return data


if __name__ == '__main__':
    main()
    