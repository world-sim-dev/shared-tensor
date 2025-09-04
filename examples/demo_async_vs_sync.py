#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
演示同步vs异步执行的差异

展示了长时间运行任务在两种模式下的表现差异
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_tensor.async_provider import async_provider

# 定义一个需要长时间运行的函数
@async_provider.share_async(name="long_running_task", wait=False)
def long_running_task(duration: int, task_name: str):
    """模拟长时间运行的任务"""
    import time
    
    print(f"🚀 开始执行 {task_name}，预计需要 {duration} 秒")
    
    for i in range(duration):
        time.sleep(1)
        progress = ((i + 1) / duration) * 100
        print(f"  📊 {task_name} 进度: {progress:.1f}%")
    
    result = f"✅ {task_name} 完成！总耗时: {duration} 秒"
    print(result)
    return result

def demo_sync_limitations():
    """演示同步方式的限制"""
    print("=" * 60)
    print("🔄 演示1: 传统同步方式的限制")
    print("=" * 60)
    
    print("❌ 问题：传统HTTP请求会因为超时而失败")
    print("   - HTTP连接超时通常为30-60秒")
    print("   - 长时间运行的任务会导致连接断开")
    print("   - 客户端无法获得执行结果")
    print("   - 服务器资源可能被浪费")
    
    print("\n💡 模拟场景：尝试执行一个需要2分钟的任务...")
    print("   (在传统同步方式下，这通常会失败)")

def demo_async_advantages():
    """演示异步方式的优势"""
    print("\n" + "=" * 60)
    print("✨ 演示2: 异步任务执行的优势")
    print("=" * 60)
    
    try:
        print("🚀 提交长时间运行的任务...")
        
        # 提交多个长时间任务
        task_ids = []
        tasks_info = [
            (15, "深度学习模型训练"),
            (10, "大数据分析"),
            (12, "图像处理算法")
        ]
        
        print("\n📋 提交任务到服务器...")
        start_time = time.time()
        
        for duration, name in tasks_info:
            task_id = long_running_task(duration, name)
            task_ids.append((task_id, name))
            print(f"  ✓ 已提交: {name} (任务ID: {task_id[:8]}...)")
        
        submit_time = time.time() - start_time
        print(f"\n⚡ 所有任务提交完成，耗时: {submit_time:.2f} 秒")
        print("💡 注意：提交过程很快，不受任务执行时间影响！")
        
        # 监控任务执行
        print(f"\n📈 监控 {len(task_ids)} 个任务的执行状态...")
        print("💡 您可以随时断开连接，稍后再来查看结果")
        
        completed_tasks = 0
        start_monitor = time.time()
        
        while completed_tasks < len(task_ids):
            print(f"\n⏰ 检查时间: {time.strftime('%H:%M:%S')}")
            
            for task_id, name in task_ids:
                try:
                    status = async_provider.get_task_status(task_id)
                    status_emoji = {
                        "pending": "⏳",
                        "running": "🔄", 
                        "completed": "✅",
                        "failed": "❌",
                        "cancelled": "🚫"
                    }
                    
                    emoji = status_emoji.get(status.status.value, "❓")
                    elapsed = time.time() - status.created_at
                    
                    print(f"  {emoji} {name}: {status.status.value} (已运行 {elapsed:.1f}s)")
                    
                    if status.status.value == "completed":
                        if not hasattr(status, '_result_printed'):
                            result = async_provider.get_task_result(task_id)
                            print(f"    💎 结果: {result}")
                            status._result_printed = True
                            completed_tasks += 1
                    
                except Exception as e:
                    print(f"  ❌ {name}: 查询状态失败 - {e}")
            
            if completed_tasks < len(task_ids):
                print("  💤 等待5秒后再次检查...")
                time.sleep(5)
        
        total_time = time.time() - start_monitor
        print(f"\n🎉 所有任务完成！总监控时间: {total_time:.1f} 秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步演示失败: {e}")
        return False

def demo_comparison():
    """对比总结"""
    print("\n" + "=" * 60)
    print("📊 同步 vs 异步 对比总结")
    print("=" * 60)
    
    print("🔄 传统同步方式:")
    print("  ❌ 受HTTP超时限制")
    print("  ❌ 客户端必须保持连接")
    print("  ❌ 网络问题会导致任务丢失")
    print("  ❌ 无法处理长时间运行的任务")
    print("  ❌ 资源利用率低")
    
    print("\n✨ 异步任务方式:")
    print("  ✅ 不受HTTP超时限制")
    print("  ✅ 客户端可以断开重连")
    print("  ✅ 任务在服务器端持续执行")
    print("  ✅ 支持任意长时间的任务")
    print("  ✅ 可以并行处理多个任务")
    print("  ✅ 支持任务监控和管理")
    print("  ✅ 更好的错误恢复机制")
    
    print("\n💡 适用场景:")
    print("  🧠 深度学习模型训练")
    print("  📊 大数据分析和处理")
    print("  🖼️  图像/视频处理")
    print("  🔬 科学计算和仿真")
    print("  📈 批量数据处理")

def main():
    """主演示函数"""
    print("🎭 Shared Tensor 异步执行演示")
    print("🔗 服务器地址: http://localhost:8080")
    print("⚠️  请确保服务器正在运行: python3 scripts/run_server.py")
    print("\n开始演示...")
    
    # 演示同步方式的限制
    demo_sync_limitations()
    
    # 演示异步方式的优势
    success = demo_async_advantages()
    
    # 对比总结
    demo_comparison()
    
    if success:
        print("\n🎉 演示完成！异步任务系统成功运行！")
        print("💡 您现在可以处理任意长时间的任务，而不用担心超时问题。")
        return 0
    else:
        print("\n❌ 演示失败，请检查服务器连接。")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
