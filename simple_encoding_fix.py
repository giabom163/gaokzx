#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的中文路径编码修复方案
不进行复杂的编码转换，而是保持原始路径并标准化格式
"""

import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK：emoji 等字符无法编码会导致 print 抛
# UnicodeEncodeError 崩溃整个流程，这里统一改为"不可编码字符替换"模式。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def create_simple_encoding_fix_patch():
    """创建简单有效的编码修复补丁"""
    
    # 动态导入 paddle.inference
    try:
        import paddle.inference
    except ImportError as e:
        print(f"[ERR] paddle.inference 导入失败: {e}")
        return False
    
    # 防止重复打补丁（重复调用会层层包裹原方法）
    if getattr(paddle.inference.Config, '_simple_encoding_patch_applied', False):
        print("[OK] 简单编码修复补丁已应用，跳过重复应用")
        return True

    print("[INFO] 创建简单有效的编码修复补丁...")
    
    # 保存原始的Config.__init__方法
    original_config_init = paddle.inference.Config.__init__
    
    def simple_patched_config_init(self, model_file, params_file=None):
        """简单的Config初始化修复 - 不改变编码，只标准化路径"""
        
        print(f"[INFO] 原始路径处理:")
        print(f"  model_file: {model_file}")
        if params_file:
            print(f"  params_file: {params_file}")
        
        try:
            # 方法：保持原始路径，不进行编码转换
            # 只做最基本的路径标准化
            
            processed_model_file = str(model_file)
            processed_params_file = str(params_file) if params_file else None
            
            # 检查是否包含中文字符
            has_chinese_model = any(ord(c) > 127 for c in processed_model_file)
            has_chinese_params = processed_params_file and any(ord(c) > 127 for c in processed_params_file)
            
            if has_chinese_model or has_chinese_params:
                # print(f"[INFO] 检测到中文路径，使用简单标准化处理...")
                
                # 标准化路径格式（只替换斜杠，不改变编码）
                processed_model_file = processed_model_file.replace('\\', '/')
                if processed_params_file:
                    processed_params_file = processed_params_file.replace('\\', '/')
                
                # print(f"  📝 标准化后的model_file: {processed_model_file}")
                # if processed_params_file:
                    # print(f"  📝 标准化后的params_file: {processed_params_file}")
                
                # 尝试转换为绝对路径（这通常能解决编码问题）
                try:
                    abs_model_path = Path(processed_model_file).resolve()
                    if abs_model_path.exists():
                        processed_model_file = str(abs_model_path)
                        # print(f"  ✅ 使用绝对路径: {processed_model_file}")
                    else:
                        print(f"  [INFO] 文件不存在，保持相对路径: {processed_model_file}")
                except Exception as e:
                    print(f"  [WARN] 绝对路径处理异常: {e}")
                    # 保持原始路径
                
                if processed_params_file:
                    try:
                        abs_params_path = Path(processed_params_file).resolve()
                        if abs_params_path.exists():
                            processed_params_file = str(abs_params_path)
                            # print(f"  ✅ 使用绝对参数路径: {processed_params_file}")
                        else:
                            print(f"  [INFO] 参数文件不存在，保持相对路径: {processed_params_file}")
                    except Exception as e:
                        print(f"  [WARN] 绝对参数路径处理异常: {e}")
                        # 保持原始路径
            
            print(f"[INFO] 最终处理后路径:")
            print(f"  model_file: {processed_model_file}")
            if processed_params_file:
                print(f"  params_file: {processed_params_file}")
            
            # 使用处理后的路径调用原始方法
            if processed_params_file:
                result = original_config_init(self, processed_model_file, processed_params_file)
            else:
                result = original_config_init(self, processed_model_file)
            
            print(f"[OK] Config初始化成功")
            return result
            
        except Exception as e:
            print(f"[ERR] 路径编码修复失败: {e}")
            print(f"[INFO] 回退到原始路径...")
            # 回退到原始路径
            if params_file:
                return original_config_init(self, model_file, params_file)
            else:
                return original_config_init(self, model_file)
    
    # 应用补丁
    paddle.inference.Config.__init__ = simple_patched_config_init
    
    # 标记补丁已应用
    paddle.inference.Config._simple_encoding_patch_applied = True
    print("[OK] 简单编码修复补丁应用成功！")
    return True


def test_simple_patch():
    """测试简单补丁效果"""
    print("\n[TEST] 测试简单编码修复补丁...")
    
    # 动态导入 paddle.inference
    try:
        import paddle.inference
    except ImportError as e:
        print(f"[ERR] paddle.inference 导入失败: {e}")
        return False
    
    # 检查是否应用了补丁
    if not hasattr(paddle.inference.Config, '_simple_encoding_patch_applied'):
        print("[ERR] 简单编码修复补丁未应用")
        return False
    
    print("[OK] 简单编码修复补丁已应用")
    return True


if __name__ == "__main__":
    # 设置中文编码环境
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    
    print("== 简单中文路径编码修复工具 ==")
    print("=" * 50)
    
    # 应用补丁
    success = create_simple_encoding_fix_patch()
    
    if success:
        # 测试补丁
        test_simple_patch()
        
        print("\n修复要点:")
        print("1. [OK] 保持原始路径编码，不进行转换")
        print("2. [OK] 标准化路径格式（斜杠统一）")
        print("3. [OK] 使用绝对路径确保路径正确性")
        print("4. [OK] 简单有效的解决方案")
        
        print("\n使用建议:")
        print("在PaddleOCR项目开头添加:")
        print("```python")
        print("import simple_encoding_fix")
        print("simple_encoding_fix.create_simple_encoding_fix_patch()")
        print("```")
    else:
        print("[ERR] 应用简单编码修复补丁失败")