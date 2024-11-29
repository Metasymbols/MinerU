import hashlib


def compute_md5(file_bytes):
    """
    计算给定字节数据的MD5值。

    使用hashlib库创建一个MD5哈希对象，然后将文件的字节数据更新到哈希对象中，
    最后返回计算出的MD5值的十六进制表示形式，转换为大写。

    参数:
    file_bytes: bytes类型，表示文件的字节数据。

    返回值:
    str类型，表示计算出的MD5值的十六进制表示形式，转换为大写。
    """
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest().upper()

def compute_sha256(input_string):
    """
    计算给定字符串的SHA-256哈希值。
    
    参数:
    input_string (str): 需要计算哈希值的输入字符串。
    
    返回:
    str: 输入字符串的SHA-256哈希值。
    """
    # 创建一个SHA-256哈希器实例
    hasher = hashlib.sha256()
    # 在Python3中，需要将字符串转化为字节对象才能被哈希函数处理
    input_bytes = input_string.encode('utf-8')
    # 更新哈希器，使其处理输入的字节对象
    hasher.update(input_bytes)
    # 返回计算得到的SHA-256哈希值的十六进制表示
    return hasher.hexdigest()