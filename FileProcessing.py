import datetime

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
        dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%d %H:%M:%S')
        if u != u_pre:
            print u_pre, len(u_session)
            if len(u_session) >= 2:
                for session in u_session:
                    for l in session:
                        fw.write(l + '\n')
            u_session = []
            session = []
            dt_pre = None
        if dt_pre is None:
            dt_pre = dt
        if (dt - dt_pre).total_seconds() / 3600.0 > hour_gap:
            if len(session) >= 2:
                u_session.append(session)
            session = []
        session.append(line.strip())
        u_pre = u
        dt_pre = dt
    if len(session) >= 2:
        u_session.append(session)
    if len(u_session) >= 2:
        for session in u_session:
            for l in session:
                fw.write(l + '\n')
    fr.close()
    fw.close()

def reverse_raw_file(file_path, save_path, u_freq_min=10, v_freq_min=15):
    u_freq = {}
    v_freq = {}
    f = open(file_path, 'r', -1)
    for line in f:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        u_freq[u] = 1 if u not in u_freq else u_freq[u] + 1
        v_freq[v] = 1 if v not in v_freq else v_freq[v] + 1
    f.close()

    fr = open(file_path, 'rt')
    fw = open(save_path, 'wt')
    lines = []
    u_pre = '0'
    for line in fr:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        if u_freq[u] < u_freq_min or v_freq[v] < v_freq_min:
            continue
        if u == u_pre:
            lines.append(line)
        else:
            for l in lines[::-1]:
                fw.write(l)
            lines = [line]
            u_pre = u
    for l in lines[::-1]:
        fw.write(l)
    fr.close()
    fw.close()