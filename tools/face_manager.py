#!/usr/bin/env micropython
"""
人脸管理工具 / Face Management Tool

用于管理人脸注册的命令行工具
Command line tool for managing face registrations
"""

import sys
import os
from modules.face_register import FaceRegister
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

def print_menu():
    """打印菜单"""
    print("\n" + "="*50)
    print("人脸管理工具 / Face Management Tool")
    print("="*50)
    print("1. 从图片注册人脸")
    print("2. 从目录批量注册")
    print("3. 从摄像头注册")
    print("4. 列出已注册人脸")
    print("5. 删除人脸注册")
    print("6. 清空所有注册")
    print("0. 退出")
    print("="*50)

def register_from_image():
    """从图片注册"""
    img_path = input("请输入图片路径: ")
    person_name = input("请输入人员姓名 (留空使用文件名): ")
    
    if not person_name:
        person_name = None
    
    register = FaceRegister()
    if register.register_from_image(img_path, person_name):
        print("注册成功！")
    else:
        print("注册失败！")
    register.deinit()

def register_from_directory():
    """从目录批量注册"""
    directory = input("请输入图片目录路径: ")
    
    register = FaceRegister()
    if register.register_from_directory(directory):
        print("批量注册完成！")
    else:
        print("批量注册失败！")
    register.deinit()

def register_from_camera():
    """从摄像头注册"""
    person_name = input("请输入要注册的人员姓名: ")
    
    if not person_name:
        print("姓名不能为空！")
        return
    
    register = FaceRegister()
    print("请面向摄像头，保持面部在画面中央...")
    if register.register_from_camera(person_name):
        print("注册成功！")
    else:
        print("注册失败！")
    register.deinit()

def list_registrations():
    """列出已注册人脸"""
    register = FaceRegister()
    registrations = register.list_registrations()
    
    if registrations:
        print("\n已注册人脸列表:")
        print("-" * 40)
        for i, reg in enumerate(registrations, 1):
            print(f"{i}. {reg['name']} - 大小: {reg['size']} bytes")
    else:
        print("没有已注册的人脸")
    
    register.deinit()

def delete_registration():
    """删除人脸注册"""
    person_name = input("请输入要删除的人员姓名: ")
    
    if not person_name:
        print("姓名不能为空！")
        return
    
    confirm = input(f"确认删除 {person_name} 的注册信息？(y/n): ")
    if confirm.lower() == 'y':
        register = FaceRegister()
        if register.delete_registration(person_name):
            print("删除成功！")
        else:
            print("删除失败！")
        register.deinit()
    else:
        print("取消删除")

def clear_all_registrations():
    """清空所有注册"""
    confirm = input("确认清空所有人脸注册信息？这将无法恢复！(yes/no): ")
    if confirm.lower() == 'yes':
        config = ConfigLoader()
        database_dir = config.get('face_recognition.database_dir')
        device_id = config.get('system.device_id', 'default')
        database_dir = database_dir + device_id + "/"
        
        try:
            if os.path.exists(database_dir):
                files = os.listdir(database_dir)
                count = 0
                for file in files:
                    if file.endswith('.bin'):
                        os.remove(database_dir + file)
                        count += 1
                print(f"已删除 {count} 个注册信息")
            else:
                print("数据库目录不存在")
        except Exception as e:
            print(f"清空失败: {e}")
    else:
        print("取消清空")

def main():
    """主函数"""
    logger = get_logger()
    logger.info("人脸管理工具启动")
    
    while True:
        print_menu()
        choice = input("\n请选择操作: ")
        
        if choice == '1':
            register_from_image()
        elif choice == '2':
            register_from_directory()
        elif choice == '3':
            register_from_camera()
        elif choice == '4':
            list_registrations()
        elif choice == '5':
            delete_registration()
        elif choice == '6':
            clear_all_registrations()
        elif choice == '0':
            print("退出程序")
            break
        else:
            print("无效的选择，请重新输入")
    
    logger.info("人脸管理工具退出")

if __name__ == "__main__":
    main()