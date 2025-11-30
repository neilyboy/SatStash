#!/usr/bin/env python3
"""
SatStash Scheduler Daemon
Runs in background and executes scheduled recordings
"""

import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sxm.utils.recorder_integration import RecorderIntegration
from sxm.utils.config import Config
from sxm.utils.session_manager import SessionManager


def load_schedules():
    """Load scheduled recordings"""
    schedule_file = Path.home() / ".seriouslyxm" / "scheduled_recordings.json"
    
    if not schedule_file.exists():
        return []
    
    with open(schedule_file) as f:
        return json.load(f)


def save_schedules(schedules):
    """Save schedules back to file"""
    schedule_file = Path.home() / ".seriouslyxm" / "scheduled_recordings.json"
    
    with open(schedule_file, 'w') as f:
        json.dump(schedules, f, indent=2)


def execute_recording(schedule):
    """Execute a scheduled recording"""
    print(f"\n{'='*60}")
    print(f"🔴 EXECUTING SCHEDULED RECORDING")
    print(f"{'='*60}")
    print(f"📻 Channel: {schedule['channel']['name']}")
    print(f"⏱️  Duration: {schedule['duration_minutes']} minutes")
    print(f"{'='*60}\n")
    
    try:
        # Initialize recorder
        config = Config()
        recorder_integration = RecorderIntegration(config)
        quality = schedule.get('audio_quality') or config.get('audio_quality', '256k')
        
        # Progress callbacks
        def progress(msg):
            print(f"  {msg}")
        
        def track_update(msg):
            print(f"\n{msg}")
        
        # Execute recording
        print(f"🎚️  Quality: {quality}")
        tracks = recorder_integration.record_channel(
            schedule['channel'],
            duration_minutes=schedule['duration_minutes'],
            quality=quality,
            progress_callback=progress,
            track_callback=track_update
        )
        
        # Update schedule status
        schedule['status'] = 'completed'
        schedule['completed_at'] = datetime.now().isoformat()
        schedule['tracks_recorded'] = len(tracks) if tracks else 0
        
        print(f"\n✅ Recording completed! Recorded {len(tracks) if tracks else 0} tracks")
        
    except Exception as e:
        print(f"\n❌ Recording failed: {e}")
        schedule['status'] = 'failed'
        schedule['error'] = str(e)
        schedule['failed_at'] = datetime.now().isoformat()


def run_scheduler():
    """Main scheduler loop"""
    print("🕐 SatStash Scheduler Started")
    print("Checking for scheduled recordings every 30 seconds...")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Load schedules
            schedules = load_schedules()
            
            if not schedules:
                time.sleep(30)
                continue
            
            # Check each pending schedule
            now = datetime.now()
            updated = False
            
            for schedule in schedules:
                if schedule['status'] != 'pending':
                    continue
                
                scheduled_time = datetime.fromisoformat(schedule['start_time'])
                
                # Check if it's time to record (within 1 minute window)
                time_diff = (scheduled_time - now).total_seconds()
                
                if -60 <= time_diff <= 60:
                    print(f"\n⏰ Time to record: {schedule['channel']['name']}")
                    execute_recording(schedule)
                    updated = True
                elif time_diff < -300:
                    # More than 5 minutes overdue - mark as missed
                    print(f"\n⚠️  Missed recording: {schedule['channel']['name']}")
                    schedule['status'] = 'missed'
                    schedule['missed_at'] = datetime.now().isoformat()
                    updated = True
                elif 0 < time_diff <= 300:
                    # Coming up in next 5 minutes
                    minutes = int(time_diff // 60)
                    print(f"⏰ Upcoming: {schedule['channel']['name']} in {minutes} minutes")
            
            # Save if any updates
            if updated:
                save_schedules(schedules)
            
            # Sleep for 30 seconds
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n👋 Scheduler stopped")
            break
        except Exception as e:
            print(f"\n❌ Scheduler error: {e}")
            time.sleep(60)


if __name__ == '__main__':
    run_scheduler()
