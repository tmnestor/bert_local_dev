from typing import Optional, Tuple
from tqdm.auto import tqdm

class ProgressManager:
    """Centralized progress bar management"""
    
    def __init__(self, disable: bool = False):
        self.disable = disable
        self.trial_bar = None
        self.epoch_bar = None
        self.batch_bar = None
        
    def init_trial_bar(self, total_trials: int) -> tqdm:
        """Initialize trial progress bar"""
        if self.trial_bar is not None:
            self.trial_bar.close()
        
        if not self.disable:
            self.trial_bar = tqdm(
                total=total_trials,
                desc="Trial",
                position=0,
                leave=True,
                ncols=80,
                bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}'
            )
        return self.trial_bar
    
    def init_epoch_bar(self, total_epochs: int, mode: str = 'train') -> tqdm:
        """Initialize epoch progress bar"""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            
        if not self.disable:
            self.epoch_bar = tqdm(
                total=total_epochs,
                desc=f"[{mode.capitalize()}: 1/{total_epochs}]",
                position=1,
                leave=False,
                ncols=100,  # Wider display
                bar_format='{desc:<20} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{postfix}]'
            )
        return self.epoch_bar
    
    def init_batch_bar(self, total_batches: int) -> tqdm:
        """Initialize batch progress bar"""
        if self.batch_bar is not None:
            self.batch_bar.close()
            
        if not self.disable:
            self.batch_bar = tqdm(
                total=total_batches,
                desc="Evaluating",
                position=0,  # Single progress bar for evaluation
                leave=False,
                ncols=100,  # Wider display
                bar_format='{desc:<12} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{postfix}]'
            )
        return self.batch_bar
    
    def update_trial(self, trial_num: int, best_score: float) -> None:
        """Update trial progress"""
        if self.trial_bar and not self.disable:
            self.trial_bar.update(1)
            self.trial_bar.set_postfix({'best_score': f'{best_score:.4f}'})
    
    def update_epoch(self, epoch: int, total_epochs: int, metrics: dict) -> None:
        """Update epoch progress"""
        if self.epoch_bar and not self.disable:
            self.epoch_bar.set_description(f"[{epoch+1}/{total_epochs}]")
            # Format metrics more cleanly
            metrics_str = ' '.join(f"{k}={v}" for k, v in metrics.items())
            self.epoch_bar.set_postfix_str(metrics_str)
            self.epoch_bar.update(1)
    
    def update_batch(self, batch_metrics: dict) -> None:
        """Update batch progress"""
        if self.batch_bar and not self.disable:
            # Format metrics more cleanly
            metrics_str = ' '.join(f"{k}={v}" for k, v in batch_metrics.items())
            self.batch_bar.set_postfix_str(metrics_str)
            self.batch_bar.update(1)
    
    def close_all(self) -> None:
        """Clean up all progress bars"""
        for bar in [self.batch_bar, self.epoch_bar, self.trial_bar]:
            if bar is not None:
                bar.close()
                print('\r', end='')  # Clear line after closing each bar
        self.batch_bar = None
        self.epoch_bar = None
        self.trial_bar = None

    def clear_output(self) -> None:
        """Clear progress bars and ensure clean output"""
        self.close_all()
        print('\r', end='')  # Clear current line
        print("")  # Add blank line for separation
