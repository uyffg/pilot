"""Full-vocabulary forward KL with bounded vocabulary-logit memory."""
import torch
from torch.utils.checkpoint import checkpoint


def _kl_chunk(student, teacher, weight):
    student_logp = torch.nn.functional.linear(student.to(weight.dtype), weight).float().log_softmax(-1)
    with torch.no_grad():
        teacher_logp = torch.nn.functional.linear(teacher.to(weight.dtype), weight).float().log_softmax(-1)
        teacher_prob = teacher_logp.exp()
    return (teacher_prob * (teacher_logp - student_logp)).sum()


def full_vocab_kl(student_hidden, teacher_hidden, weight, chunk_size=16):
    if student_hidden.shape != teacher_hidden.shape or not student_hidden.shape[0]:
        raise ValueError('Teacher and student must have the same nonempty action span')
    if weight.requires_grad:
        raise ValueError('This bounded-memory projection requires a frozen output head')
    teacher_hidden = teacher_hidden.detach()
    loss = student_hidden.new_zeros((), dtype=torch.float32)
    # Capture the immutable head instead of passing it as a checkpoint input.
    # With activation offload, explicit inputs would copy this large weight to
    # CPU once per chunk. Only the small hidden-state inputs should be saved.
    def project_chunk(student, teacher):
        return _kl_chunk(student, teacher, weight)
    for start in range(0, student_hidden.shape[0], chunk_size):
        student = student_hidden[start:start + chunk_size]
        teacher = teacher_hidden[start:start + chunk_size]
        # Recompute vocabulary probabilities during backward; retain hidden
        # states, not an action_length x vocabulary graph.
        loss = loss + checkpoint(project_chunk, student, teacher, use_reentrant=False)
    return loss / student_hidden.shape[0]
