import xml.etree.ElementTree as ET

def dot2polygon(xml_path, lymphocyte_half_box_size, monocytes_half_box_size, min_spacing, output_path):
    '''
    :param xml_path (str): the path of the annotation file, ex. root\sub_root\filename.xml
    :param lymphocyte_half_box_size (folat): the size of half of the bbox around the lymphocyte dot in um, 4.5 for lymphocyte
    :param monocytes_half_box_size (folat): the size of half of the bbox around the monocytes dot in um, 11.0 for monocytes
    :param min_spacing (float): the minimum spacing of the wsi corresponding to the annotations
    :param output_path (str): the output path
    :return:
    '''


    # parsing the annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lymphocyte_half_box_size = lymphocyte_half_box_size / min_spacing
    monocytes_half_box_size = monocytes_half_box_size/min_spacing

    # iterating through the dot annotation.
    for A in root.iter('Annotation'):

        #Lymphocytes:
        if (A.get('PartOfGroup')=="lymphocytes") & (A.get('Type')=="Dot"):
        # change the type to Polygon
            A.attrib['Type'] = "Polygon"

            for child in A:
                for sub_child in child:
                    x_value = sub_child.attrib['X']
                    y_value = sub_child.attrib['Y']
                    sub_child.attrib['X'] = str(float(sub_child.attrib['X'])-lymphocyte_half_box_size)
                    sub_child.attrib['Y'] = str(float(sub_child.attrib['Y'])-lymphocyte_half_box_size)
                child.append(ET.Element(sub_child.tag, Order = '1', X=str(float(x_value)-lymphocyte_half_box_size), Y=str(float(y_value)+lymphocyte_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='2', X=str(float(x_value)+lymphocyte_half_box_size), Y=str(float(y_value)+lymphocyte_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='3', X=str(float(x_value)+lymphocyte_half_box_size), Y=str(float(y_value)-lymphocyte_half_box_size) ))


        # Monoocytes:
        if (A.get('PartOfGroup')=="monocytes") & (A.get('Type')=="Dot"):
        # change the type to Polygon
            A.attrib['Type'] = "Polygon"

            for child in A:
                for sub_child in child:
                    x_value = sub_child.attrib['X']
                    y_value = sub_child.attrib['Y']
                    sub_child.attrib['X'] = str(float(sub_child.attrib['X'])-monocytes_half_box_size)
                    sub_child.attrib['Y'] = str(float(sub_child.attrib['Y'])-monocytes_half_box_size)
                child.append(ET.Element(sub_child.tag, Order = '1', X=str(float(x_value)-monocytes_half_box_size), Y=str(float(y_value)+monocytes_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='2', X=str(float(x_value)+monocytes_half_box_size), Y=str(float(y_value)+monocytes_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='3', X=str(float(x_value)+monocytes_half_box_size), Y=str(float(y_value)-monocytes_half_box_size) ))



    # writing the new annotation file
    tree.write(output_path)



import xml.etree.ElementTree as ET
import math

def dot2polygon_segmentation(xml_path, 
                             lymphocyte_radius_um, 
                             monocytes_radius_um, 
                             min_spacing, 
                             output_path,
                             num_vertices=32):
    '''
    :param xml_path (str): the path of the annotation file, ex. root\sub_root\filename.xml
    :param lymphocyte_radius_um (float): the radius in um for the lymphocyte polygon
    :param monocytes_radius_um (float): the radius in um for the monocytes polygon
    :param min_spacing (float): the minimum spacing of the wsi corresponding to the annotations (um/pixel)
    :param output_path (str): the output path
    :param num_vertices (int): number of vertices to approximate the circle
    :return:
    '''

    # parsing the annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Convert from micrometer to pixel based on min_spacing
    lymphocyte_radius_px = lymphocyte_radius_um / min_spacing
    monocytes_radius_px = monocytes_radius_um / min_spacing

    for A in root.iter('Annotation'):
        part_of_group = A.get('PartOfGroup')
        ann_type = A.get('Type')

        # 対象がlymphocytesまたはmonocytesで、TypeがDotの場合に処理
        if ann_type == "Dot" and (part_of_group == "lymphocytes" or part_of_group == "monocytes"):
            A.attrib['Type'] = "Polygon"  # DotからPolygonに変更

            # radiusを対象物によって選択
            if part_of_group == "lymphocytes":
                radius_px = lymphocyte_radius_px
            else:  # monocytes
                radius_px = monocytes_radius_px

            # AnnotationのCoordinatesを取得
            # Dotの場合、Coordinates要素は1点のみ持っている想定
            for child in A:
                coords = list(child)  # Coordinate要素のリスト
                if len(coords) == 1:
                    # 元のドットの中心座標
                    x_center = float(coords[0].attrib['X'])
                    y_center = float(coords[0].attrib['Y'])

                    # 既存のCoordinate要素を一旦削除
                    for c in coords:
                        child.remove(c)

                    # 円を近似する多角形頂点を追加
                    for i in range(num_vertices):
                        theta = 2 * math.pi * i / num_vertices
                        x_vertex = x_center + radius_px * math.cos(theta)
                        y_vertex = y_center + radius_px * math.sin(theta)
                        # Orderは頂点番号、整数値でOK
                        vertex = ET.Element('Coordinate',
                                            Order=str(i),
                                            X=str(x_vertex),
                                            Y=str(y_vertex))
                        child.append(vertex)
                else:
                    # ドットアノテーションが1点以外を持つ想定は通常ないが、
                    # 念のため何もしない
                    pass

    # writing the new annotation file
    tree.write(output_path)
    print(f"Converted dot annotations to polygon annotations and saved to {output_path}")
