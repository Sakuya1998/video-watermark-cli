import sys
import multiprocessing
import warnings

if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')


class CommonPool:
    """普通模式下的任务池"""
    def map(self, func, args):
        return list(map(func, args))


class AutoPool:
    """自动选择多线程或多进程的任务池"""
    def __init__(self, mode='common', processes=1):
        """
        初始化自动任务池
        Args:
            mode (str): 模式（'multithreading', 'multiprocessing', 'vectorization', 'cached', 'common'）
            processes (int): 线程或进程数量
        """
        if mode == 'multiprocessing' and sys.platform == 'win32':
            warnings.warn('multiprocessing not supported in windows, turning to multithreading')
            mode = 'multithreading'

        self.mode = mode
        self.processes = processes

        if mode == 'vectorization':
            # 向量化模式（此处暂未实现具体逻辑）
            self.pool = None
        elif mode == 'cached':
            # 缓存模式（此处暂未实现具体逻辑）
            self.pool = None
        elif mode == 'multithreading':
            # 多线程模式
            from multiprocessing.dummy import Pool as ThreadPool
            try:
                self.pool = ThreadPool(processes=processes)
            except Exception as e:
                raise RuntimeError(f"Failed to create ThreadPool: {e}")
        elif mode == 'multiprocessing':
            # 多进程模式（非Windows平台）
            from multiprocessing import Pool
            try:
                self.pool = Pool(processes=processes)
            except Exception as e:
                raise RuntimeError(f"Failed to create Pool: {e}")
        else:  # common
            # 普通模式
            self.pool = CommonPool()

    def map(self, func, args):
        """
        执行任务映射
        Args:
            func (function): 要执行的函数
            args (list): 参数列表
        Returns:
            list: 执行结果列表
        """
        if self.mode in ['multithreading', 'multiprocessing']:
            try:
                return self.pool.map(func, args)
            except Exception as e:
                raise RuntimeError(f"Task execution failed: {e}")
        else:
            return self.pool.map(func, args)