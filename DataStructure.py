import string
import datetime
import numpy as np
import pickle
import math
hour_gap = 6
test_portion = 0.2
test_session_len_min = 2

class DataLoader(object):
    def __init__(self, hour_gap=6):
        self.hour_gap = hour_gap
        self.u_uid = {}
        self.uid_u = {}
        self.v_vid = {}
        self.w_wid = {}
        self.uid_records = {}
        self.nu = 0
        self.nv = 0
        self.nt = 24 * 7
        self.nw = 0
        self.nr = 0
        self.vid_coor = {}
        self.vid_coor_rad = {}
        self.vid_coor_nor = {}
        self.vid_coor_nor_rectified = {}
        self.vid_pop = {}
        self.sampling_list = []

    def add_records(self, file_path, dl_save_path, glove_path, glove_save_path, u_cnt_max=-1, blacklist=None):
        f = open(file_path, 'r', -1)
        for line in f:
            al = line.strip().split('\t')
            u = al[0]
            if u in blacklist:
                continue
            v = al[4]
            dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%d %H:%M:%S')
            lat = string.atof(al[2])
            lng = string.atof(al[3])
            if u not in self.u_uid:
                if u_cnt_max > 0 and len(self.u_uid) >= u_cnt_max:
                    break
                print u, self.nu
                self.u_uid[u] = self.nu
                self.uid_u[self.nu] = u
                self.uid_records[self.nu] = UserRecords(self.nu)
                self.nu += 1
            if v not in self.v_vid:
                self.v_vid[v] = self.nv
                self.vid_pop[self.nv] = 0
                self.vid_coor_rad[self.nv] = np.array([np.radians(lat), np.radians(lng)])
                self.vid_coor[self.nv] = np.array([lat, lng])
                self.nv += 1
            wids = []
            if len(al) == 6:
                w_al = al[5].split(' ')
                for w in w_al:
                    if w not in self.w_wid:
                        self.w_wid[w] = self.nw
                        self.nw += 1
                    wids.append(self.w_wid[w])
            uid = self.u_uid[u]
            vid = self.v_vid[v]
            self.sampling_list.append(vid)
            self.vid_pop[vid] += 1
            self.uid_records[uid].add_record(dt, uid, vid, wids)
            self.nr += 1
        f.close()

        coor_mean = np.zeros(2)
        coor_var = np.zeros(2)
        for vid, coor in self.vid_coor.items():
            coor_mean += coor
        coor_mean /= len(self.vid_coor)
        for vid, coor in self.vid_coor.items():
            coor_var += (coor - coor_mean) ** 2
        coor_var /= len(self.vid_coor)
        coor_var = np.sqrt(coor_var)
        for vid in self.vid_coor:
            self.vid_coor_nor[vid] = (self.vid_coor[vid] - coor_mean) / coor_var
            lat_sub = self.vid_coor[vid][0] - coor_mean[0]
            lng_sub = self.vid_coor[vid][1] - coor_mean[1]
            lat_rectified = lat_sub / coor_var[0]
            lng_rectified = lng_sub * math.cos(self.vid_coor[vid][0]) / coor_var[0]
            self.vid_coor_nor_rectified[vid] = np.array([lat_rectified, lng_rectified])

        pickle.dump(self, open(dl_save_path, 'wb'))

        glove_file = open(glove_path, 'rt', -1)
        glove_save_file = open(glove_save_path, 'wt', -1)
        for line in glove_file:
            idx = line.index(' ')
            word = line[0: idx]
            if word in self.w_wid:
                wid = self.w_wid[word]
                glove_save_file.write(str(wid) + '\t' + line[idx + 1: -1] + '\n')
        glove_file.close()
        glove_save_file.close()

    def show_info(self):
        print 'U: ', self.nu, 'V: ', self.nv, 'R: ', self.nr, 'W: ', self.nw

class Record(object):
    def __init__(self, dt, uid, vid, wids, vid_next=-1, tid_next = -1, is_first=False, is_last=False):
        self.dt = dt
        self.uid = uid
        self.vid = vid
        self.tid = dt.hour
        if dt.weekday > 4:
            self.tid += 24
        self.tid_168 = dt.weekday() * 24 + dt.hour
        self.wids = [] if wids is None else wids
        self.vid_next = vid_next
        self.tid_next = tid_next
        self.is_first = is_first
        self.is_last = is_last


    def peek(self):
        print 'u: ', self.uid, '\tv: ', self.vid, '\tt: ', self.tid, '\twids: ', self.wids, \
            '\tvid_next: ', self.vid_next, '\tis_first: ', self.is_first, '\tis_last: ', self.is_last


class UserRecords(object):
    def __init__(self, uid):
        self.uid = uid
        self.records = []
        self.session_cnt = 0
        self.session_start_idx=[]
        self.dt_last = None
        self.test_idx = 0
        self.record_cnt_train = 0
        self.record_cnt_test = 0

    def add_record(self, dt, uid, vid, wids):
        is_first = False
        if self.dt_last is None or (dt - self.dt_last).total_seconds() / 3600.0 > hour_gap:
            self.session_start_idx.append(len(self.records))
            self.session_cnt += 1
            self.test_idx = self.session_start_idx[int(self.session_cnt * (1 - test_portion))]
            is_first = True
            if len(self.records) > 0:
                self.records[len(self.records) - 1].is_last = True
        record = Record(dt, uid, vid, wids, is_first=is_first, is_last=True)
        if len(self.records) > 0:
            self.records[len(self.records) - 1].vid_next = record.vid
            self.records[len(self.records) - 1].tid_next = record.tid
            if not is_first:
                self.records[len(self.records) - 1].is_last = False
            else:
                self.records[len(self.records) - 1].vid_next = -1
        self.records.append(record)
        self.dt_last = dt

    def get_records(self, mod=0):
        if mod == 0:  # train only
            return self.records[0: self.test_idx]
        elif mod == 1:  # test only
            return self.records[self.test_idx: len(self.records)]
        else:
            return self.records

    def get_predicting_records_cnt(self, mod=0):
        cnt = 0
        if mod == 0:  # train only
            for record in self.records[0: self.test_idx]:
                if record.is_last or record.is_first:
                    continue
                cnt += 1
            return cnt
        else:  # test only
            for record in self.records[self.test_idx: len(self.records)]:
                if record.is_last or record.is_first:
                    continue
                cnt += 1
            return cnt
