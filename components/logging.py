"""
TensorBoard Logging Utilities
=============================

Clean logging interface for training metrics.
"""

from typing import Dict, Optional, Any
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """
    Clean interface for logging training metrics to TensorBoard.
    
    Usage:
        logger = MetricsLogger(writer)
        logger.log_step(global_step, loss_metrics, aux_info, config)
        logger.log_epoch(epoch, train_metrics, eval_metrics)
    """
    
    def __init__(self, writer: Optional[SummaryWriter] = None):
        self.writer = writer
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        aux: Dict[str, Any],
        config: Any,
        cell_acc: float = 0.0,
        task_acc: float = 0.0,
    ):
        """Log per-step metrics to TensorBoard."""
        if self.writer is None:
            return
        
        # === Core Metrics ===
        self._log_losses(step, metrics)
        self._log_accuracy(step, cell_acc, task_acc)
        self._log_compute_stats(step, aux, config)
        self._log_gates(step, aux, metrics)
        self._log_memory_stats(step, aux)
        self._log_confidence(step, aux)
        self._log_entropy(step, aux)
        self._log_q_halt(step, aux, metrics)
        self._log_temperature(step, aux, config)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_cell_acc: float,
        train_task_acc: float,
        eval_cell_acc: Optional[float] = None,
        eval_task_acc: Optional[float] = None,
    ):
        """Log per-epoch metrics to TensorBoard."""
        if self.writer is None:
            return
        
        self.writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('Epoch/train_cell_acc', train_cell_acc, epoch)
        self.writer.add_scalar('Epoch/train_task_acc', train_task_acc, epoch)
        
        if eval_cell_acc is not None:
            self.writer.add_scalar('Epoch/eval_cell_acc', eval_cell_acc, epoch)
        if eval_task_acc is not None:
            self.writer.add_scalar('Epoch/eval_task_acc', eval_task_acc, epoch)
    
    def log_image(self, tag: str, image, step: int):
        """Log image to TensorBoard."""
        if self.writer is not None:
            self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close the writer."""
        if self.writer is not None:
            self.writer.close()
    
    # === Private Logging Methods ===
    
    def _log_losses(self, step: int, metrics: Dict[str, float]):
        """Log loss components."""
        # Core losses
        self._log('Loss/total', metrics.get('loss_total'), step)
        self._log('Loss/task', metrics.get('loss_task'), step)
        
        # Component losses
        self._log('Loss/q_halt', metrics.get('loss_q_halt'), step)
        self._log('Loss/gate_polarization', metrics.get('loss_gate_polar'), step)
        self._log('Loss/feedback_polarization', metrics.get('loss_feedback_polar'), step)
        self._log('Loss/gate_sparsity', metrics.get('loss_gate_sparsity'), step)
        self._log('Loss/layer_divergence', metrics.get('loss_layer_divergence'), step)
        self._log('Loss/model_info_gain', metrics.get('loss_model_info_gain'), step)
        self._log('Loss/diversity', metrics.get('loss_diversity'), step)
        self._log('Loss/confidence_calibration', metrics.get('loss_confidence_calibration'), step)
        self._log('Loss/ponder', metrics.get('loss_ponder_adaptive'), step)
        
        # Model info gain (positive = good, means info is flowing)
        self._log('InfoGain/model_avg', metrics.get('avg_model_info_gain'), step)
    
    def _log_accuracy(self, step: int, cell_acc: float, task_acc: float):
        """Log accuracy metrics."""
        self._log('Metrics/cell_accuracy', cell_acc, step)
        self._log('Metrics/task_accuracy', task_acc, step)
    
    def _log_compute_stats(self, step: int, aux: Dict, config: Any):
        """Log compute utilization statistics."""
        passes = aux.get('passes_run', 1)
        max_passes = getattr(config, 'max_passes', passes)
        
        self._log('Compute/model_passes', passes, step)
        self._log('Compute/pass_utilization', passes / max_passes, step)
        
        layer_iters = aux.get('layer_iterations', [])
        if layer_iters:
            avg_iters = sum(layer_iters) / len(layer_iters)
            total_steps = sum(layer_iters)
            self._log('Compute/avg_layer_iterations', avg_iters, step)
            self._log('Compute/total_layer_steps', total_steps, step)
        
        # Layer-level entropy and info gain
        layer_entropies = aux.get('layer_entropies', [])
        if layer_entropies:
            for i, ent in enumerate(layer_entropies):
                val = ent.mean().item() if hasattr(ent, 'mean') else ent
                self._log(f'LayerEntropy/iter_{i}', val, step)
        
        layer_info_gains = aux.get('layer_info_gains', [])
        if layer_info_gains:
            for i, ig in enumerate(layer_info_gains):
                val = ig.mean().item() if hasattr(ig, 'mean') else ig
                self._log(f'LayerInfoGain/iter_{i+1}', val, step)
            # Also log total absolute info gain (measure of layer activity)
            total_abs_ig = sum(abs(ig.mean().item() if hasattr(ig, 'mean') else ig) for ig in layer_info_gains)
            self._log('LayerInfoGain/total_abs', total_abs_ig, step)
    
    def _log_gates(self, step: int, aux: Dict, metrics: Dict[str, float]):
        """Log gate statistics."""
        # Memory gates (from aux)
        if aux.get('read_gate_count', 0) > 0:
            avg_read = aux['read_gate_sum'] / aux['read_gate_count']
            self._log('Gates/read_mean', avg_read, step)
        
        if aux.get('write_gate_count', 0) > 0:
            avg_write = aux['write_gate_sum'] / aux['write_gate_count']
            self._log('Gates/write_mean', avg_write, step)
        
        # Feedback gates
        if aux.get('answer_feedback_count', 0) > 0:
            avg_fb = aux['answer_feedback_sum'] / aux['answer_feedback_count']
            self._log('Feedback/answer_gate_mean', avg_fb, step)
        
        if aux.get('iteration_feedback_count', 0) > 0:
            avg_fb = aux['iteration_feedback_sum'] / aux['iteration_feedback_count']
            self._log('Feedback/iteration_gate_mean', avg_fb, step)

    def _log_memory_stats(self, step: int, aux: Dict):
        """Log memory usage statistics."""
        # Read/write rates as percentages (0-100%)
        self._log('Memory/LTM_read_rate_pct', aux.get('ltm_read_rate'), step)
        self._log('Memory/LTM_write_rate_pct', aux.get('ltm_write_rate'), step)
        self._log('Memory/WM_read_rate_pct', aux.get('wm_read_rate'), step)
        self._log('Memory/WM_write_rate_pct', aux.get('wm_write_rate'), step)
        # WM fullness metrics
        self._log('Memory/WM_slots_active_pct', aux.get('wm_slots_active_pct'), step)
        self._log('Memory/WM_avg_freshness_pct', aux.get('wm_avg_freshness_pct'), step)
    
    def _log_confidence(self, step: int, aux: Dict):
        """Log confidence tracking."""
        pass_confs = aux.get('pass_confidences', [])
        if pass_confs:
            for i, conf in enumerate(pass_confs):
                val = conf.mean().item() if hasattr(conf, 'mean') else conf
                self._log(f'Confidence/pass_{i}', val, step)
            final = pass_confs[-1]
            self._log('Confidence/final', final.mean().item() if hasattr(final, 'mean') else final, step)
    
    def _log_entropy(self, step: int, aux: Dict):
        """Log entropy and information gain."""
        # Entropy per pass
        pass_entropies = aux.get('pass_entropies', [])
        if pass_entropies:
            for i, ent in enumerate(pass_entropies):
                self._log(f'Entropy/pass_{i}', ent, step)
            self._log('Entropy/final', pass_entropies[-1], step)
        
        # Information gain (may be tensors for gradient flow, detach before .item())
        pass_info_gains = aux.get('pass_info_gains', [])
        if pass_info_gains:
            for i, ig in enumerate(pass_info_gains):
                # Detach first to avoid warning about .item() on gradient tensor
                if hasattr(ig, 'detach'):
                    ig_val = ig.detach().item()
                elif hasattr(ig, 'item'):
                    ig_val = ig.item()
                else:
                    ig_val = ig
                self._log(f'InfoGain/pass_{i+1}', ig_val, step)
            final_ig = pass_info_gains[-1]
            if hasattr(final_ig, 'detach'):
                final_ig_val = final_ig.detach().item()
            elif hasattr(final_ig, 'item'):
                final_ig_val = final_ig.item()
            else:
                final_ig_val = final_ig
            self._log('InfoGain/final', final_ig_val, step)
    
    def _log_q_halt(self, step: int, aux: Dict, metrics: Dict[str, float]):
        """Log Q-halt metrics."""
        # Q-halt values per pass
        pass_q_halt = aux.get('pass_q_halt', [])
        if pass_q_halt:
            for i, qh in enumerate(pass_q_halt):
                self._log(f'QHalt/pass_{i}', qh, step)
            self._log('QHalt/final', pass_q_halt[-1], step)
            
            # Track when model would halt (q_halt > 0)
            would_halt = [1 if qh > 0 else 0 for qh in pass_q_halt]
            halt_pass = next((i for i, h in enumerate(would_halt) if h), len(would_halt))
            self._log('QHalt/would_halt_at_pass', halt_pass, step)
        
        # Halt reason: 0=max_passes, 1=q_halt_correct, 2=stuck_no_progress
        halt_reason = aux.get('halt_reason', 'max_passes')
        halt_reason_map = {'max_passes': 0, 'q_halt_correct': 1, 'stuck_no_progress': 2}
        self._log('QHalt/halt_reason', halt_reason_map.get(halt_reason, 0), step)
        
        # Q-halt accuracy from metrics
        self._log('QHalt/accuracy', metrics.get('q_halt_accuracy'), step)
    
    def _log_temperature(self, step: int, aux: Dict, config: Any):
        """Log Gumbel-softmax temperature."""
        temp = aux.get('temperature')
        if temp is None and hasattr(config, 'get_temperature'):
            temp = config.get_temperature(step)
        self._log('Training/temperature', temp, step)
    
    def _log(self, tag: str, value, step: int):
        """Helper to log a single scalar (skips None values)."""
        if value is not None and self.writer is not None:
            self.writer.add_scalar(tag, value, step)


def create_logger(log_dir) -> MetricsLogger:
    """Create a MetricsLogger with a SummaryWriter."""
    writer = SummaryWriter(log_dir)
    return MetricsLogger(writer)
