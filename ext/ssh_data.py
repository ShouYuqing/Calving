"""
get data from VM
"""
from paramiko import SSHClient
from scp import SCPClient

port = 22
hostname = "40.112.218.137"
username = "hs"
password = "Shihan960324!"
dst = "."
src = "/home/hs/date/calve_data.json"

ssh = SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.load_system_host_keys()
ssh.connect(hostname=hostname, port=22, username=username, password=password,
            pkey=None, key_filename=None, timeout=None, allow_agent=True,
            look_for_keys=True, compress=False)

with SCPClient(ssh.get_transport(), sanitize=lambda x: x) as scp:
    scp.get(remote_path=src, local_path=dst)

scp.close()