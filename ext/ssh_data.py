"""
get data from VM
"""
import paramiko
from scp import SCPClient

def ssh_get(dst = "../data/", src = "/home/hs/date/calve_data.json", port = 22, hostname = "168.61.55.8", username = "cloud", password = "cloud12345678!"):
    """
    get data through ssh
    :param dst: destination dir
    :param src: data src
    :return:
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    ssh.connect(hostname=hostname, port=port, username=username, password=password,
                pkey=None, key_filename=None, timeout=None, allow_agent=True,
                look_for_keys=True, compress=False)

    with SCPClient(ssh.get_transport(), sanitize=lambda x: x) as scp:
        scp.get(remote_path=src, local_path=dst)

    scp.close()

def ssh_send(dst = "../data/predict_result.json", src = "/home/cloud/TEMP_FRONT_END/build/", port = 22, hostname = "104.41.132.238", username = "cloud", password = "cloud123456!"):
    """
    send data to the VM
    :param dst: dst
    :param src: src
    :param port: port
    :param hostname: hname
    :param username: uname
    :param password: pasw
    :return:
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    ssh.connect(hostname=hostname, port=port, username=username, password=password,
                pkey=None, key_filename=None, timeout=None, allow_agent=True,
                look_for_keys=True, compress=False)

    with SCPClient(ssh.get_transport(), sanitize=lambda x: x) as scp:
        scp.put(dst, src)

    scp.close()

if __name__ == "__main__":
    # get training data
    #ssh_get(src = "-r /home/cloud/date/training_data")

    # get predict data
    #ssh_get(src = "-r /home/cloud/predict_data1")
    #ssh_get(src = "-r /home/cloud/predict_data2")

    # send data
    ssh_send()
