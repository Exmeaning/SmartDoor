"""
K230 智能开门系统启动脚本 / K230 Smart Door Boot Script

系统启动时自动执行 / Auto-execute on system boot
"""

import sys
import gc
import uos

def setup_system():
    """系统设置 / System setup"""
    
    # 设置系统路径
    sys.path.append('/sdcard')
    sys.path.append('/sdcard/libs')
    
    # 打印系统信息
    print("\n" + "="*50)
    print("K230 Smart Door Lock System")
    print("="*50)
    
    # 打印系统信息
    info = uos.uname()
    print(f"System: {info.sysname}")
    print(f"Release: {info.release}")
    print(f"Version: {info.version}")
    print(f"Machine: {info.machine}")
    
    # 内存信息
    print(f"Memory: {gc.mem_alloc()} used, {gc.mem_free()} free")
    
    # 垃圾回收
    gc.collect()
    
    print("Boot sequence completed")
    print("="*50 + "\n")

# 执行系统设置
setup_system()

# 自动运行主程序
try:
    import main
    main.main()
except Exception as e:
    print(f"Failed to start main program: {e}")