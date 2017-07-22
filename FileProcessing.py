import datetime
import string
def generate_session_file(file_path, save_path, hour_gap=6):
    fr = open(file_path, 'rt', -1)
    fw = open(save_path, 'wt')
    u_session = []
    session = []
    u_pre = 'start'
    dt_pre = None
    u_set = set()
    for line in fr:
        al = line.split('\t')
        u = al[0]
        u_set.add(u)
        # if len(u_set) == 1000:
        #     break
        dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%dT%H:%M:%SZ')
        if u != u_pre:
            # print u_pre, len(u_session)
            if len(u_session) >= 2:
                session_valid_cnt = 0
                for session in u_session:
                    if len(session) >= 2:
                        session_valid_cnt += 1
                if session_valid_cnt >= 2:
                    for session in u_session:
                        for l in session:
                            fw.write(l + '\n')
            u_session = []
            session = []
            dt_pre = None
        if dt_pre is None:
            dt_pre = dt
        if (dt - dt_pre).total_seconds() / 3600.0 > hour_gap:
            # if len(session) >= 2:
            u_session.append(session)
            session = []
        session.append(line.strip())
        u_pre = u
        dt_pre = dt
    u_session.append(session)
    if len(u_session) >= 2:
        session_valid_cnt = 0
        for session in u_session:
            if len(session) >= 2:
                session_valid_cnt += 1
        if session_valid_cnt >= 2:
            for session in u_session:
                for l in session:
                    fw.write(l + '\n')
    fr.close()
    fw.close()

def process_raw_file(file_path, save_path, u_freq_min=10, u_vcnt_min=10, lat_min=-180, lat_max=-180, lng_min=180, lng_max=180):
    u_freq = {}
    f = open(file_path, 'r', -1)
    for line in f:
        al = line.strip().split('\t')
        u = al[0]
        u_freq[u] = 1 if u not in u_freq else u_freq[u] + 1
    f.close()
    u_set = set()
    fr = open(file_path, 'rt')
    fw = open(save_path, 'wt')
    lines = []
    v_set = set()
    u_pre = '0'
    for line in fr:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        lat = string.atof(al[2])
        lng = string.atof(al[3])
        if lat < lat_min or lat > lat_max or lng < lng_min or lng > lng_max:
            continue
        if u_freq[u] < u_freq_min:
            continue
        if u == u_pre:
            lines.append(line)
            v_set.add(v)
        else:
            u_set.add(u)
            if len(v_set) >= u_vcnt_min:
                for l in lines[::-1]:
                    fw.write(l)
            v_set = set()
            v_set.add(v)
            lines = [line]
            u_pre = u
    u_set.add(u)
    if len(v_set) >= u_vcnt_min:
        for l in lines[::-1]:
            fw.write(l)
    fr.close()
    fw.close()
    print 'U: ', len(u_set)

# def reverse_raw_file(file_path, save_path, u_freq_min=10, v_freq_min=15):
#     u_freq = {}
#     v_freq = {}
#     f = open(file_path, 'r', -1)
#     for line in f:
#         al = line.strip().split('\t')
#         if len(al) < 5:
#             print line
#             raw_input()
#         u = al[0]
#         v = al[4]
#         u_freq[u] = 1 if u not in u_freq else u_freq[u] + 1
#         v_freq[v] = 1 if v not in v_freq else v_freq[v] + 1
#     f.close()
#
#     fr = open(file_path, 'rt')
#     fw = open(save_path, 'wt')
#     lines = []
#     u_pre = '0'
#     for line in fr:
#         al = line.strip().split('\t')
#         u = al[0]
#         v = al[4]
#         if u_freq[u] < u_freq_min or v_freq[v] < v_freq_min:
#             continue
#         if u == u_pre:
#             lines.append(line)
#         else:
#             for l in lines[::-1]:
#                 fw.write(l)
#             lines = [line]
#             u_pre = u
#     for l in lines[::-1]:
#         fw.write(l)
#     fr.close()
#     fw.close()