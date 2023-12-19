import torch


def focal_loss(
    prob: torch.Tensor,
    true: torch.Tensor,
    gamma: float = 3,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Focal loss 값을 계산함

    Args:
        prob: predicted value (prob)
        true: label (ground truth)
        gamma: reduction hyper-parameter
        alpha: Weighting factor in range (0,1) to balance
               positive vs negative examples or -1 for ignore. Default = 0.25
               if alpha = 0.75 -> more sensitive to positive sample
               than alpha = 0.25

               Example)
               if alpha=0.75 & positive: 0.75 * 1 + (1 - 0.75) * (1-1) = 0.75
               if alpha=0.75 & negative: 0.75 * 1 + (1 - 0.75) * (1-0) = 0.25

               if alpha=0.25 & positive: 0.25 * 1 + (1 - 0.25) * (1-1) = 0.25
               if alpha=0.25 & negative: 0.25 * 1 + (1 - 0.25) * (1-0) = 1.0

    Return:
        focal_loss (torch.Tensor): focal loss

    See Also:
        - Original paper: https://arxiv.org/abs/1708.02002
        - Blog: https://woochan-autobiography.tistory.com/929

    Note:
        Equation (5) in original paper
            FL(pt) = -\alpha_{t}(1 - p_{t})** \gamma  * log(p_{t})
            - t: 클래스 번호
            - alpha_{t}: weighted (balanced) constant
            - p_{t}: model confidence
            - (1 - p_{t})** \gamma: modulating factor. confidence가 큰 경우에 gamma만
            큼 제곱하여 reduction해주는 기능
            - log(p_{t}): cross-entropy

    """
    # pred = (n, )
    p_t = prob * true + (1 - prob) * (1 - true)
    modulating_factor = (1 - p_t) ** gamma
    ce_loss = torch.nn.functional.binary_cross_entropy(prob, true)

    if alpha > 0:
        alpha_t = alpha * true + (1 - alpha) * (1 - true)
        return alpha_t * modulating_factor * ce_loss

    return modulating_factor * ce_loss


def focal_with_bce(
    pred_bag_logit: torch.Tensor,
    true_bag_label: torch.Tensor,
    pred_instance_prob: torch.Tensor,
    true_instance_label: torch.Tensor,
    alpha=0.999,
    gamma=3,
    bag_loss_weight: float = 3.5,
    instance_loss_weight=5e3,
):
    """Bagloss (BCE) Instance loss(focal loss)을 더한 loss을 계산

    Args:
        pred_bag_label (torch.Tensor): 예측된 bag label
        true_bag_label (torch.Tensor): ground truth bag label
        pred_instance_prob (torch.Tensor): 예측된 instance의 확률
        true_instance_label (torch.Tensor): ground truth instance label
        bag_loss_weight (float, optional): bag loss 가중치. Defaults to 3.5.

    Returns:
        total_loss (torch.Tensor): BCE + focal loss
    """
    bag_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_bag_logit, true_bag_label)
    instance_loss = focal_loss(pred_instance_prob, true_instance_label, gamma=gamma, alpha=alpha)
    return bag_loss_weight * bag_loss + true_bag_label * instance_loss_weight * instance_loss.mean(
        dim=0
    )


def ranknet_loss(y_pred: torch.Tensor, y_true: torch.Tensor, sigma=1.0) -> torch.Tensor:
    """RankNet의 손실 함수 값을 계산

    See Also:
        https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf

    Args:
        y_pred (torch.Tensor): instance probablity
        y_true (torch.Tensor): instance label
        sigma (float, optional): scale parameter of sigmoid func. Defaults to 1.0.

    Returns:
        loss: rank loss in [0, 1]

    Note:
        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        >>> diff_matrix =  y_pred.view(-1, 1) - y_pred.view(1, -1)
        tensor(
            [[ 0.0000,  0.2000,  0.5000,  0.6000,  0.6000,  0.6000],
            [-0.2000,  0.0000,  0.3000,  0.4000,  0.4000,  0.4000],
            [-0.5000, -0.3000,  0.0000,  0.1000,  0.1000,  0.1000],
            [-0.6000, -0.4000, -0.1000,  0.0000,  0.0000,  0.0000],
            [-0.6000, -0.4000, -0.1000,  0.0000,  0.0000,  0.0000],
            [-0.6000, -0.4000, -0.1000,  0.0000,  0.0000,  0.0000]]
        )

        >>> y_mat = y_true.view(-1, 1)
        >>> label_diff = y_mat - y_mat.t()
        >>> pbar = torch.where(
                label_diff > 0,
                torch.tensor(1.0),
                torch.where(label_diff == 0, torch.tensor(0.5), torch.tensor(0.0))
            )
        Tensor(
            [[0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]
        )


    Example:
        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> ranknet_loss(y_pred, y_true)  tensor(0.2239)

        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        >>> ranknet_loss(y_pred, y_true)  tensor(0.2572)

        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        >>> ranknet_loss(y_pred, y_true)  tensor(0.3239)

        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        >>> ranknet_loss(y_pred, y_true)  tensor(0.2878)
    """

    # i,j을 브로드 케스팅해서 score의 차이값을 계산
    diff_matrix = y_pred.view(-1, 1) - y_pred.view(1, -1)
    pij = torch.sigmoid(-sigma * diff_matrix)

    # Y True와 차이를 계산하여 Pbar을 구함
    label_diff = y_true.view(-1, 1) - y_true.view(1, -1)
    pbar = torch.where(
        label_diff > 0,
        torch.tensor(1.0),
        torch.where(label_diff == 0, torch.tensor(0.5), torch.tensor(0.0)),
    )

    # C을 구함
    c = -pbar * torch.log(pij) - (1 - pbar) * torch.log(1 - pij)

    return 1 - torch.mean(c)


def pointwise_ranknet_loss(y_pred: torch.Tensor, y_true: torch.Tensor, sigma=1.0) -> torch.Tensor:
    """y_true인 경우의 한해서만 RankNet의 손실 함수 값을 계산

    See Also:
        https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf

    Note:
        우리문제와 일반적인 ReSys의 문제의 가장 큰 차이점은 우리 문제는 Relevance score가
        1에 밖에 없으며, 해당하는 원인변이가 1개 있고, 스코어도 1개 있음.
        그러다보니, RankNet이 쓸데없이 모든 i,j에 대해서 ground truth격인 P_ij
        을 계산을 요구함. 즉, 원인변이가 아닌 것들 중에서도 랭크의 비교 예측치인
        \bar{P_{ij}}을 계산하는데, 각 i,j을 굳이 비교할 필요가 없음.
          예를 들어, 200개의 변이중에서 1개만 원인변이이고, 199개는 원인변이가 아니다보니,
        의 200 by 200 원소를 굳이 다 합칠 필요가 없음. 따라서, label이 인 경우,
        의 행만 계산하면되지않나 생각됨.  i가 2개인 경우도 i,j 모든 행렬에 대해서
        평가하는 것이 아닌, 두 행에 대해서만 적용.

    Args:
        y_pred (torch.Tensor): instance probablity
        y_true (torch.Tensor): instance label
        sigma (float, optional): scale parameter of sigmoid func. Defaults to 1.0.

    Returns:
        loss: rank loss in [0, 1]

    Example:
        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        >>> pbar = torch.where(
                        label_diff > 0,
                        torch.tensor(1.0),
                        torch.where(label_diff == 0, torch.tensor(0.5), torch.tensor(0.0))
                    )
        Tensor(
            [[0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],  // 원인변이 Pbar 행
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]
        )

    """

    # i,j을 브로드 케스팅해서 score의 차이값을 계산
    diff_matrix = y_pred.view(-1, 1) - y_pred.view(1, -1)
    pij = torch.sigmoid(-sigma * diff_matrix)

    # Y True와 차이를 계산하여 Pbar을 구함
    label_diff = y_true.view(-1, 1) - y_true.view(1, -1)
    input_device = y_pred.device
    pbar = torch.where(
        label_diff > 0,
        torch.tensor(1.0).to(input_device),
        torch.where(
            label_diff == 0,
            torch.tensor(0.5).to(input_device),
            torch.tensor(0.0).to(input_device),
        ),
    )

    indices = torch.nonzero(pbar == 1.0).squeeze()

    pbar = pbar[indices]
    pij = pij[indices]

    # C을 구함
    c = -pbar * torch.log(pij) - (1 - pbar) * torch.log(1 - pij)
    return 1 - c.mean()
