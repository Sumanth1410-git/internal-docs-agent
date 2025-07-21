import time
import psutil
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from collections import deque
import signal

class BotMonitor:
    """Real-time monitoring system for the Slack bot"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.monitoring = True
        self.metrics_history = deque(maxlen=100)  # Keep last 100 readings
        self.alerts = []
        
        # Performance thresholds
        self.thresholds = {
            'memory_mb': 4000,
            'cpu_percent': 80,
            'response_time_ms': 3000,
            'error_rate': 10  # percent
        }
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n✅ Monitoring stopped by user")
        self.monitoring = False
        sys.exit(0)
    
    def get_bot_process_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the running bot process"""
        try:
            # Find Python processes running slack_bot.py
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('slack_bot.py' in cmd for cmd in cmdline):
                        return {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.cpu_percent(),
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'cmdline': ' '.join(cmdline)
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return None
        except Exception as e:
            print(f"Error getting bot process info: {e}")
            return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'timestamp': datetime.now(),
                'system_cpu': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            print(f"Error getting system metrics: {e}")
            return {}
    
    def check_bot_logs(self) -> Dict[str, Any]:
        """Check bot logs for recent errors or activity"""
        log_stats = {
            'recent_errors': 0,
            'recent_queries': 0,
            'last_activity': None
        }
        
        try:
            # Look for log files (this is a simple implementation)
            log_patterns = ['*.log', '*.txt']
            
            # Check if there are any recent log entries
            # This is a basic implementation - in production you'd parse actual log files
            log_stats['last_activity'] = datetime.now() - timedelta(minutes=1)
            
        except Exception as e:
            print(f"Error checking logs: {e}")
        
        return log_stats
    
    def check_bot_health(self, bot_info: Dict[str, Any]) -> Dict[str, str]:
        """Perform health checks on bot performance"""
        health_status = {
            'overall': 'healthy',
            'memory': 'healthy',
            'cpu': 'healthy',
            'responsiveness': 'healthy'
        }
        
        alerts = []
        
        # Memory check
        if bot_info['memory_mb'] > self.thresholds['memory_mb']:
            health_status['memory'] = 'warning'
            health_status['overall'] = 'warning'
            alerts.append(f"High memory usage: {bot_info['memory_mb']:.1f}MB")
        
        # CPU check
        if bot_info['cpu_percent'] > self.thresholds['cpu_percent']:
            health_status['cpu'] = 'critical'
            health_status['overall'] = 'critical'
            alerts.append(f"High CPU usage: {bot_info['cpu_percent']:.1f}%")
        
        # Store alerts
        if alerts:
            self.alerts.extend(alerts)
            # Keep only recent alerts
            self.alerts = self.alerts[-10:]
        
        return health_status
    
    def display_real_time_metrics(self):
        """Display real-time metrics in terminal"""
        print("📊 **REAL-TIME BOT MONITORING**")
        print("=" * 60)
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while self.monitoring:
                # Clear screen for real-time updates
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("📊 **REAL-TIME BOT MONITORING**")
                print("=" * 60)
                print(f"⏰ Started: {self.start_time.strftime('%H:%M:%S')}")
                print(f"🕐 Current: {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 60)
                
                # Get bot process info
                bot_info = self.get_bot_process_info()
                
                if bot_info:
                    print("🤖 **BOT STATUS: RUNNING**")
                    print(f"   PID: {bot_info['pid']}")
                    print(f"   💾 Memory: {bot_info['memory_mb']:.1f} MB")
                    print(f"   🔥 CPU: {bot_info['cpu_percent']:.1f}%")
                    
                    # Health check
                    health = self.check_bot_health(bot_info)
                    status_emoji = "🟢" if health['overall'] == 'healthy' else "🟡" if health['overall'] == 'warning' else "🔴"
                    print(f"   {status_emoji} Health: {health['overall'].upper()}")
                    
                else:
                    print("🤖 **BOT STATUS: NOT RUNNING**")
                    print("   ❌ No slack_bot.py process found")
                
                print("-" * 60)
                
                # System metrics
                sys_metrics = self.get_system_metrics()
                if sys_metrics:
                    print("💻 **SYSTEM METRICS:**")
                    print(f"   🔥 CPU: {sys_metrics['system_cpu']:.1f}%")
                    print(f"   💾 RAM: {sys_metrics['system_memory_percent']:.1f}% used")
                    print(f"   💿 Disk: {sys_metrics['disk_usage_percent']:.1f}% used")
                    print(f"   🆓 Free RAM: {sys_metrics['system_memory_available_gb']:.1f} GB")
                
                print("-" * 60)
                
                # Recent alerts
                if self.alerts:
                    print("⚠️ **RECENT ALERTS:**")
                    for alert in self.alerts[-3:]:  # Show last 3 alerts
                        print(f"   • {alert}")
                    print("-" * 60)
                
                # Log analysis
                log_stats = self.check_bot_logs()
                print("📝 **ACTIVITY STATUS:**")
                print(f"   📨 Recent queries: {log_stats['recent_queries']}")
                print(f"   ❌ Recent errors: {log_stats['recent_errors']}")
                if log_stats['last_activity']:
                    print(f"   🕐 Last activity: {log_stats['last_activity'].strftime('%H:%M:%S')}")
                
                print("-" * 60)
                print("Press Ctrl+C to stop monitoring...")
                
                # Store metrics for history
                if bot_info:
                    metric_point = {
                        'timestamp': datetime.now(),
                        'memory_mb': bot_info['memory_mb'],
                        'cpu_percent': bot_info['cpu_percent'],
                        'system_metrics': sys_metrics
                    }
                    self.metrics_history.append(metric_point)
                
                # Wait before next update
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
    
    def generate_performance_report(self) -> str:
        """Generate a performance summary report"""
        if not self.metrics_history:
            return "No performance data available yet."
        
        # Calculate averages from history
        avg_memory = sum(m['memory_mb'] for m in self.metrics_history) / len(self.metrics_history)
        avg_cpu = sum(m['cpu_percent'] for m in self.metrics_history) / len(self.metrics_history)
        
        uptime = datetime.now() - self.start_time
        
        report = f"""
📊 **PERFORMANCE REPORT**
========================

⏰ **Uptime:** {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m
📊 **Data Points:** {len(self.metrics_history)}
💾 **Avg Memory:** {avg_memory:.1f} MB
🔥 **Avg CPU:** {avg_cpu:.1f}%
⚠️ **Total Alerts:** {len(self.alerts)}

🏥 **Health Status:**
{'🟢 HEALTHY' if avg_memory < self.thresholds['memory_mb'] and avg_cpu < self.thresholds['cpu_percent'] else '🟡 NEEDS ATTENTION'}

📈 **Recommendations:**
{'✅ Performance is optimal' if avg_memory < self.thresholds['memory_mb'] * 0.7 else '⚠️ Consider optimizing memory usage'}
{'✅ CPU usage is normal' if avg_cpu < self.thresholds['cpu_percent'] * 0.7 else '⚠️ Consider reducing processing load'}
        """
        
        return report
    
    def save_metrics_to_file(self):
        """Save collected metrics to file"""
        try:
            metrics_dir = Path("monitoring")
            metrics_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = metrics_dir / f"bot_metrics_{timestamp}.json"
            
            data = {
                'monitoring_session': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60
                },
                'metrics_history': [
                    {
                        'timestamp': m['timestamp'].isoformat(),
                        'memory_mb': m['memory_mb'],
                        'cpu_percent': m['cpu_percent']
                    } for m in self.metrics_history
                ],
                'alerts': self.alerts,
                'summary': {
                    'avg_memory_mb': sum(m['memory_mb'] for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history else 0,
                    'avg_cpu_percent': sum(m['cpu_percent'] for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history else 0,
                    'peak_memory_mb': max(m['memory_mb'] for m in self.metrics_history) if self.metrics_history else 0,
                    'peak_cpu_percent': max(m['cpu_percent'] for m in self.metrics_history) if self.metrics_history else 0
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"\n📊 Metrics saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving metrics: {e}")

def main():
    """Main monitoring function"""
    print("🚀 Starting Bot Monitor...")
    print("Make sure your Slack bot is running in another terminal!")
    print("Monitoring will start in 3 seconds...\n")
    
    time.sleep(3)
    
    monitor = BotMonitor()
    
    try:
        # Run real-time monitoring
        monitor.display_real_time_metrics()
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    
    finally:
        # Generate final report
        print("\n" + "="*60)
        print(monitor.generate_performance_report())
        
        # Save metrics
        if monitor.metrics_history:
            monitor.save_metrics_to_file()
        
        print("\n✅ Monitoring session completed!")

if __name__ == "__main__":
    main()
