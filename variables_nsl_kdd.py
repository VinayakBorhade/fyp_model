categorical_columns=['protocol_type', 'service', 'flag']

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","extra"]

col_names_realtime_test = ["num_conn","startTimet","orig_pt","resp_pt1","orig_ht","resp_ht",
"duration","protocol_type","resp_pt2","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urg","hot","num_failed_logins",
"logged_in","num_compromised","root_shell","su_attempted",
"num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_hot_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate_sec","rerror_rate","srv_error_rate_sec","same_srv_rate","diff_srv_rate",
"dst_host_diff_srv_rate","count_100","srv_count_100","same_srv_rate_100","diff_srv_rate_100",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rat","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate"]

selected_col_names = ["protocol_type", "service", "flag", "src_bytes", 
"dst_bytes", "wrong_fragment", "hot", "logged_in", 
"root_shell", "num_root", "count", "srv_count", "serror_rate", 
"rerror_rate", "same_srv_rate", "diff_srv_rate", "dst_host_count", 
"dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
"dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
"dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

dum_cols = ['Protocol_type_icmp',
 'Protocol_type_tcp',
 'Protocol_type_udp',
 'service_IRC',
 'service_X11',
 'service_Z39_50',
 'service_aol',
 'service_auth',
 'service_bgp',
 'service_courier',
 'service_csnet_ns',
 'service_ctf',
 'service_daytime',
 'service_discard',
 'service_domain',
 'service_domain_u',
 'service_echo',
 'service_eco_i',
 'service_ecr_i',
 'service_efs',
 'service_exec',
 'service_finger',
 'service_ftp',
 'service_ftp_data',
 'service_gopher',
 'service_harvest',
 'service_hostnames',
 'service_http',
 'service_http_2784',
 'service_http_443',
 'service_http_8001',
 'service_imap4',
 'service_iso_tsap',
 'service_klogin',
 'service_kshell',
 'service_ldap',
 'service_link',
 'service_login',
 'service_mtp',
 'service_name',
 'service_netbios_dgm',
 'service_netbios_ns',
 'service_netbios_ssn',
 'service_netstat',
 'service_nnsp',
 'service_nntp',
 'service_ntp_u',
 'service_other',
 'service_pm_dump',
 'service_pop_2',
 'service_pop_3',
 'service_printer',
 'service_private',
 'service_red_i',
 'service_remote_job',
 'service_rje',
 'service_shell',
 'service_smtp',
 'service_sql_net',
 'service_ssh',
 'service_sunrpc',
 'service_supdup',
 'service_systat',
 'service_telnet',
 'service_tftp_u',
 'service_tim_i',
 'service_time',
 'service_urh_i',
 'service_urp_i',
 'service_uucp',
 'service_uucp_path',
 'service_vmnet',
 'service_whois',
 'flag_OTH',
 'flag_REJ',
 'flag_RSTO',
 'flag_RSTOS0',
 'flag_RSTR',
 'flag_S0',
 'flag_S1',
 'flag_S2',
 'flag_S3',
 'flag_SF',
 'flag_SH']