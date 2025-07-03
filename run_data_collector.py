#!/usr/bin/env python3
"""
Simple script to run the market data collection system.

Usage:
    python run_data_collector.py [--single] [--continuous] [--stats]
    
Options:
    --single      Run a single collection cycle
    --continuous  Run continuous data collection
    --stats       Show system statistics
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path setup
try:
    from market_data_collector import orchestrator
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


async def run_single_cycle():
    """Run a single data collection cycle"""
    logger.info("Running single data collection cycle...")
    results = await orchestrator.run_collection_cycle()
    
    logger.info("Collection Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    
    return results


async def run_continuous():
    """Run continuous data collection"""
    logger.info("Starting continuous data collection...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        await orchestrator.run_continuous()
    except KeyboardInterrupt:
        logger.info("Stopping continuous collection...")
        orchestrator.stop()


async def show_statistics():
    """Show system statistics"""
    logger.info("Generating system statistics...")
    stats = await orchestrator.generate_statistics()
    
    logger.info("System Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Market Data Collection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--single',
        action='store_true',
        help='Run a single collection cycle'
    )
    group.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuous data collection'
    )
    group.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics'
    )
    
    args = parser.parse_args()
    
    try:
        if args.single:
            await run_single_cycle()
        elif args.continuous:
            await run_continuous()
        elif args.stats:
            await show_statistics()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
