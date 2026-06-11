import os


def extract_label_txt(txt_path: str, use_player_id_as_key: bool = False):
    """
    if use_player_id_as_key:
        use player id as labels(dict)'s primary key.
    else:
        use frame id as labels(dict)'s primary key.
    """
    
    assert os.path.isfile(txt_path), f"Please provide legel file path. {txt_path} not found"

    cls_mapper = {
        '0': 0,
        '1': 1,
        '2': 0,
        '3': 2, 
        '4': 3,
        '5': 0 
    }
    cls_names = {
        0: 'none',
        1: 'idle',
        2: 'kick', 
        3: 'fall'
    }

    # === read txt lines === 
    with open(txt_path, 'r') as ftxt:
        data = ftxt.read().split('\n')

    labels = {}
    for line in data:
        if line == '':
            continue # skip empty line
        info = line.split(',')

        # === extract info ===
        frame_id = info[0]
        player_id = info[1]
        player_bbox = [float(x) for x in info[2:6]] # x0, y0, w, h
        conf = float(info[6])

        lb_cls = 0 # 沒有label的歸類在0
        if len(info) == 11:
            if info[-1] in cls_mapper: # 1 3 4
                lb_cls = cls_mapper[info[-1]] #如果有label進行轉換
            # else:
                # print(f'What?? {txt_path}, :have the label of {info[-1]}')
        lb_name = cls_names[lb_cls]
        # = cls_names[lb_cls]

        # === save info into dict === 
        k1 = player_id if use_player_id_as_key else frame_id
        k2 = frame_id if use_player_id_as_key else player_id
        if k1 not in labels:
            labels[k1] = {} # initial this key
        
        labels[k1][k2] = {
                'bbox': player_bbox,
                'bbox_conf': conf,
                'label': lb_cls,
                'label_name': lb_name
            }
    
    return labels

        