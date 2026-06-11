import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

from utils.io import extract_label_txt

class DatasetAnalysisor(object):

    """
    This analysis object deal with
        * bbox analysis
            - total w, h mean, std
            - different game's w, h mean and std
            - in one game, different clip's w, h mean and std
        * clip length # no need?
        * action length
        * action relation (sequence)
        * action class number
    """

    def __init__(self, save_path: str = "./analysis_report/"):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        self.labels = {}
        self.dataset_infos = {
            'bbox': {}, 
            'clip_length': {},
            'action_length': {},
            'action_ralation': {},
            'action_class': {}
        }


    def new_labels(self, game_id, clip_id, labels: dict):
        """
        must set use_player_id_as_key = True
        """
        if game_id not in self.labels:
            self.labels[game_id] = {}
        self.labels[game_id][clip_id] = labels


    def extract_clip_lb_info(self, labels, k, idx=None):

        def extract(labels, k1, k2, idx):
            info = labels[k1][k2][k]
            if idx is None:
                return info
            elif isinstance(idx, list):
                infos = []
                for i in idx:
                    infos.append(info[i])
                return infos
            elif isinstance(idx, int):  
                return info[idx]
            else:
                raise ValuseError('Please provide none, list or int for idx.')
            return info

        extracted_infos = []
        for k1 in labels:
            for k2 in labels[k1]:
                _info = extract(labels, k1, k2, idx)
                extracted_infos.append(_info)

        return extracted_infos


    def plot_distribution(self, x, sub_folder='', save_name='default.jpg'):

        figure_save_folder = f'{self.save_path}/{sub_folder}/'
        if not os.path.isdir(figure_save_folder):
            os.mkdir(figure_save_folder)

        plt.cla()
        plt.clf()
        N, bins, patches = plt.hist(x, bins=100)
        m = x.mean()
        plt.plot([m, m], [0, max(N)], 'r--', label='mean')
        plt.legend()
        plt.savefig(figure_save_folder + save_name)


    def analysis(self):

        """
        Part1 Bbox Shape Analysis
        """

        def build_w_h_area_array(whs):
            area, w, h = [], [], []
            for wh in whs:
                w.append(wh[0])
                h.append(wh[1])
                area.append(wh[0] * wh[1])
            area, w, h = np.array(area), np.array(w), np.array(h)
            return w, h, area

        def build_sub_msg(name, _msg, _x):
            _msg += f'    [{name}]\n'
            _msg += f'           mean: {_x.mean()}\n'
            _msg += f'           std: {_x.std()}\n'
            _msg += f'           max: {_x.max()}\n'
            _msg += f'           min: {_x.min()}\n'
            return _msg

        def build_msg(title, _a, _w, _h):
            _msg = f''
            _msg += f'{title}\n'
            _msg = build_sub_msg('area', _msg, _a)
            _msg = build_sub_msg('width', _msg, _w)
            _msg = build_sub_msg('height', _msg, _h)
            return _msg

        total_wh = []
        games_wh = {}
        clips_wh = {}
        for game_id in self.labels:
            games_wh[game_id] = []
            clips_wh[game_id] = {}
            for clip_id in self.labels[game_id]:
                clip_wh = self.extract_clip_lb_info(self.labels[game_id][clip_id], 
                                                    k = 'bbox', 
                                                    idx = [2, 3])
                # total w, h mean, std
                total_wh += clip_wh

                # different game's w, h mean and std
                games_wh[game_id] += clip_wh
                
                # in one game, different clip's w, h mean and std
                clips_wh[game_id][clip_id] = clip_wh

        msg = f''
        # analysis total_wh
        w, h, area = build_w_h_area_array(total_wh)
        self.plot_distribution(area, 'total_bbox', 'total_bbox_area.jpg')
        self.plot_distribution(w, 'total_bbox', 'total_bbox_width.jpg')
        self.plot_distribution(h, 'total_bbox', 'total_bbox_height.jpg')
        msg += build_msg('==== total bbox ====', area, w, h)

        # analysis different game's inner
        games_w, games_h, games_area = [], [], []
        for game_id in games_wh:
            w, h, area = build_w_h_area_array(games_wh[game_id])
            msg += build_msg(f'\n==== game-id: {game_id} ====', area, w, h)
            # self.plot_distribution(area, f'games_bbox/{game_id}', 'area.jpg')
            # self.plot_distribution(w, f'games_bbox/{game_id}', 'width.jpg')
            # self.plot_distribution(h, f'games_bbox/{game_id}', 'height.jpg')
            games_w.append(w.mean())
            games_h.append(h.mean())
            games_area.append(area.mean())

        games_w, games_h, games_area = \
            np.array(games_w), np.array(games_h), np.array(games_area)

        msg += build_msg('\n==== between games bbox ====', games_area, games_w, games_h)

        # analysis every clips
        # for game_id in clips_wh:
        #     for clip_id in clips_wh[game_id]:
        #         w, h, area = build_w_h_area_array(games_wh[game_id])
        #         msg += build_msg(f'==== game-id: {game_id}, clip-id: {clip_id} ====', area, w, h)


        with open(f'{self.save_path}/bbox_analysis.txt', 'w') as ftxt:
            ftxt.write(msg)


        """
        Part2 Bbox Motion, Player Action Length Analysis
        """

        transition_mat = np.zeros([4, 4])
        max_diff, max_diff_info = [], []
        max_diff_ratio, max_diff_ratio_info = [], []

        def motion(game_id, clip_id, labels, max_diff, max_diff_info, max_diff_ratio, max_diff_ratio_info):
            player_lengths = []
            action_lengths = []
            player_w_diff = []
            player_h_diff = []
            for player_id in labels:
                # player length
                player_lengths.append(len(labels[player_id]))
                count = 0
                prvs_lb = None
                prvs_h, prvs_w = None, None
                max_frame_id = max([int(x) for x in labels[player_id]])
                for frame_id in range(max_frame_id + 1):
                    frame_id = str(frame_id)
                    
                    if frame_id not in labels[player_id]:
                        prvs_lb = None
                        prvs_h, prvs_w = None, None
                        continue

                    # player action length
                    now_label = labels[player_id][frame_id]['label']
                    if prvs_lb is None or now_label != prvs_lb:
                        # and build transition matrix
                        if prvs_lb is not None:
                            transition_mat[prvs_lb][now_label] += 1
                        action_lengths.append(count)
                        count = 0
                    count += 1
                    prvs_lb = now_label

                    w, h = labels[player_id][frame_id]['bbox'][2:4]
                    wh_ratio = w / h
                    diff_h, diff_w = None, None
                    if prvs_h is not None:
                        diff_h = abs(h - prvs_h)
                        player_h_diff.append(diff_h)
                    if prvs_w is not None:
                        diff_w = abs(w - prvs_w)
                        player_w_diff.append(diff_w)

                    if prvs_w is not None and prvs_h is not None:
                        wh_ratio_prvs = prvs_w / prvs_h

                    prvs_h = h
                    prvs_w = w

                    if diff_w is not None and diff_h is not None:
                        diff = (diff_w ** 2 + diff_h ** 2) ** 0.5
                        diff_ratio = abs(wh_ratio_prvs - wh_ratio)
                        if len(max_diff) < 10:
                            max_diff.append(diff)
                            max_diff_info.append([game_id, clip_id, player_id, frame_id])
                            if len(max_diff) == 10:
                                # find min and argmin
                                min_idx = max_diff.index(min(max_diff))
                                # swap
                                tmp1 = max_diff[min_idx]
                                tmp2 = max_diff_info[min_idx]
                                max_diff[min_idx] = max_diff[-1]
                                max_diff_info[min_idx] = max_diff_info[-1]
                                max_diff[-1] = tmp1
                                max_diff_info[-1] = tmp2

                        elif diff >= max_diff[-1]:
                            del max_diff[-1]
                            max_diff.append(diff)
                            max_diff_info.append([game_id, clip_id, player_id, frame_id])
                            # find min and argmin
                            min_idx = max_diff.index(min(max_diff))
                            # swap
                            tmp1 = max_diff[min_idx]
                            tmp2 = max_diff_info[min_idx]
                            max_diff[min_idx] = max_diff[-1]
                            max_diff_info[min_idx] = max_diff_info[-1]
                            max_diff[-1] = tmp1
                            max_diff_info[-1] = tmp2

                        if len(max_diff_ratio) < 10:
                            max_diff_ratio.append(diff_ratio)
                            max_diff_ratio_info.append([game_id, clip_id, player_id, frame_id])
                            if len(max_diff_ratio) == 10:
                                # find min and argmin
                                min_idx = max_diff_ratio.index(min(max_diff_ratio))
                                # swap
                                tmp1 = max_diff_ratio[min_idx]
                                tmp2 = max_diff_ratio_info[min_idx]
                                max_diff_ratio[min_idx] = max_diff_ratio[-1]
                                max_diff_ratio_info[min_idx] = max_diff_ratio_info[-1]
                                max_diff_ratio[-1] = tmp1
                                max_diff_ratio_info[-1] = tmp2

                        elif diff >= max_diff_ratio[-1]:
                            del max_diff_ratio[-1]
                            max_diff_ratio.append(diff)
                            max_diff_ratio_info.append([game_id, clip_id, player_id, frame_id])
                            # find min and argmin
                            min_idx = max_diff_ratio.index(min(max_diff_ratio))
                            # swap
                            tmp1 = max_diff_ratio[min_idx]
                            tmp2 = max_diff_ratio_info[min_idx]
                            max_diff_ratio[min_idx] = max_diff_ratio[-1]
                            max_diff_ratio_info[min_idx] = max_diff_ratio_info[-1]
                            max_diff_ratio[-1] = tmp1
                            max_diff_ratio_info[-1] = tmp2

            return player_lengths, action_lengths, player_w_diff, player_h_diff, max_diff, max_diff_info, max_diff_ratio, max_diff_ratio_info


        total_player_lengths, total_action_lengths, total_player_w_diff, total_player_h_diff = \
            [], [], [], []
        for game_id in self.labels:
            for clip_id in self.labels[game_id]:
                res = motion(game_id, clip_id, self.labels[game_id][clip_id], max_diff, max_diff_info, max_diff_ratio, max_diff_ratio_info)
                total_player_lengths += res[0]
                total_action_lengths += res[1]
                total_player_w_diff += res[2]
                total_player_h_diff += res[3]
                max_diff, max_diff_info = res[4], res[5]
                max_diff_ratio, max_diff_ratio_info = res[6], res[7]

        total_player_lengths = np.array(total_player_lengths)
        total_action_lengths = np.array(total_action_lengths)
        total_player_w_diff = np.array(total_player_w_diff)
        total_player_h_diff = np.array(total_player_h_diff)

        self.plot_distribution(total_player_lengths, 'total_bbox_motion', 'total_player_lengths.jpg')
        self.plot_distribution(total_action_lengths, 'total_bbox_motion', 'total_action_lengths.jpg')
        self.plot_distribution(total_player_w_diff, 'total_bbox_motion', 'total_player_w_diff.jpg')
        self.plot_distribution(total_player_h_diff, 'total_bbox_motion', 'total_player_h_diff.jpg')
        msg2 = f''
        msg2 += f' ==== Total Motion ==== \n'
        msg2 = build_sub_msg('player_lengths', msg2, total_player_lengths)
        msg2 = build_sub_msg('action_lengths', msg2, total_action_lengths)
        msg2 = build_sub_msg('player_w_diff', msg2, total_player_w_diff)
        msg2 = build_sub_msg('player_h_diff', msg2, total_player_h_diff)

        plt.cla()
        plt.clf()
        plt.matshow(transition_mat)
        for (i, j), val in np.ndenumerate(transition_mat):
            plt.text(j, i, f'{val:.0f}', ha='center', va='center', color='white')
        plt.savefig(f'{self.save_path}/action_transition.jpg')

        with open(f'{self.save_path}/bbox_motion_analysis.txt', 'w') as ftxt:
            ftxt.write(msg2)

        msg3 = f'Different Analysis\n'
        for i in range(len(max_diff)):
            msg3 += f'i-th: {i}\n'
            msg3 += f'max different: {max_diff[i]}\n'
            game_id, clip_id, player_id, frame_id = max_diff_info[i]
            msg3 += f'at game: {game_id}\n'
            msg3 += f'   clip: {clip_id}\n'
            msg3 += f'   player_id: {player_id}\n'
            msg3 += f'   frame_id: {frame_id}\n'
            msg3 += f'\n'

        with open(f'{self.save_path}/max_diff_info.txt', 'w') as ftxt:
            ftxt.write(msg3)

        msg4 = f'Different Analysis\n'
        for i in range(len(max_diff_ratio)):
            msg4 += f'i-th: {i}\n'
            msg4 += f'max different: {max_diff_ratio[i]}\n'
            game_id, clip_id, player_id, frame_id = max_diff_ratio_info[i]
            msg4 += f'at game: {game_id}\n'
            msg4 += f'   clip: {clip_id}\n'
            msg4 += f'   player_id: {player_id}\n'
            msg4 += f'   frame_id: {frame_id}\n'
            msg4 += f'\n'

        with open(f'{self.save_path}/max_diff_ratio_info.txt', 'w') as ftxt:
            ftxt.write(msg4)


    def analysis_dataset(self, root: str):
        """
        root/
            game1/
                1/ # contain images
                1.txt
                2/ # contain images
                2.txt
                ...
            game2/
            ...
        """
        print(f'Reading all labels in dataset: {root}')
        start_time = time.time()

        games = os.listdir(root)
        for game in games:
            game_path = f'{root}/{game}/'
            files_and_folders = os.listdir(game_path)
            txt_files = [x for x in files_and_folders if x.split('.')[-1] == 'txt']
            for txt in txt_files:
                txt_path = game_path + txt
                _label = extract_label_txt(txt_path, use_player_id_as_key=True)
                self.new_labels(game, txt.split('.')[0], _label)

        print(f'done. cost: {time.time() - start_time} sec\n')

        
        print(f'Analysis bbox information')
        start_time = time.time()
        self.analysis()
        print(f'done. cost: {time.time() - start_time} sec\n')
