from loguru import logger


def ImportPIL(f):
    """
    尝试导入PIL模块，如果失败则记录错误信息并退出程序。

    此函数作为一个装饰器，用于确保在调用被装饰的函数之前，
    环境中已经安装了Pillow库。如果没有安装，会记录一条错误日志，
    并退出程序，状态码为1。

    参数:
    - f: 被装饰的函数。

    返回:
    - f: 原封不动地返回被装饰的函数。
    """
    try:
        import PIL  # noqa: F401
    except ImportError:
        logger.error('Pillow not installed, please install by pip.')
        exit(1)
    return f
