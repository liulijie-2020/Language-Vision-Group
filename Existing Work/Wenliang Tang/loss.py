# 省略前面部分的代码....

@registry.register_loss("ocr_clip")
class ocrclipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.0])

    def forward_ce_clip(self, sample_list, model_output):
        loss_mask = model_output["clip_mask"]
        clip_scores_mask_process = model_output["clip_scores"]*loss_mask.unsqueeze(1)*loss_mask.unsqueeze(-1)
        clip_scores = clip_scores_mask_process
        clip_scores1 = clip_scores_mask_process.transpose(-1, -2)

        clip_tgt = torch.arange(0, clip_scores.size(1), 1).repeat(clip_scores.size(0), 1).to(clip_scores.device)
        #clip_tgt = clip_tgt
        clip_score_length = loss_mask.sum(dim=-1).cpu()
        clip_score_length += (clip_score_length == 0).float()
        pack_scores = pack_padded_sequence(clip_scores, clip_score_length, batch_first=True, enforce_sorted=False).data
        pack_scores1 = pack_padded_sequence(clip_scores1, clip_score_length, batch_first=True, enforce_sorted=False).data
        pack_targets = pack_padded_sequence(clip_tgt, clip_score_length, batch_first=True, enforce_sorted=False).data

        loss_ce1 = F.cross_entropy(pack_scores, pack_targets)
        loss_ce2 = F.cross_entropy(pack_scores1, pack_targets)
        losses = (loss_ce1+loss_ce2)/2

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        clip_loss = torch.sum(losses) / count
        return clip_loss

    def forward_clip(self, sample_list, model_output):
        clip_scores = model_output["clip_scores"]
        tgt_nums = clip_scores.size(1)

        clip_tgt = torch.zeros(tgt_nums, tgt_nums).scatter_(1, torch.arange(0, tgt_nums, 1).unsqueeze(1), 1). \
            repeat(clip_scores.size(0), 1, 1).to(clip_scores.device)
        # clip_tgt size: [batch_size, tgt_nums, tgt_nums], and each clip_tgt[i] = eyes(tgt_nums, tgt_nums)
        loss_mask = model_output["clip_mask"]

        losses1 = F.binary_cross_entropy_with_logits(clip_scores, clip_tgt, reduction="none")
        losses1 *= loss_mask.unsqueeze(-1)
        # losses1 *= loss_mask
        # losses2 *= loss_mask

        losses = losses1
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        clip_loss = torch.sum(losses) / count
        return clip_loss

    def forward_large_clip(self, sample_list, model_output):
        clip_scores = model_output["clip_scores"]
        tgt_nums = clip_scores.size(0)
        clip_tgt = torch.zeros(tgt_nums, tgt_nums).scatter_(1, torch.arange(0, tgt_nums, 1).unsqueeze(1),1 ).to(clip_scores.device)

        loss_mask = model_output["clip_mask"]
        loss_mask_row = loss_mask.repeat(1,loss_mask.size(0))
        loss_mask_col = loss_mask.transpose(-1, -2).repeat(loss_mask.size(0), 1)

        losses1 = F.binary_cross_entropy_with_logits(clip_scores, clip_tgt, reduction="none")
        losses1 *= (loss_mask_row*loss_mask_col)
        # losses1 *= loss_mask
        # losses2 *= loss_mask

        losses = losses1
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        clip_loss = torch.sum(losses) / count
        return clip_loss

    def forward(self, sample_list, model_output):
        eta_ratio = 1.0
        loss = (eta_ratio * self.forward_clip(sample_list, model_output))

        return loss
 

# 省略后面的代码...
