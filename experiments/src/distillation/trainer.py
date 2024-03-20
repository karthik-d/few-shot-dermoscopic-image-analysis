"""
ADAPTED FROM https://github.com/HobbitLong/RepDistiller
"""

def run_concrete_train_loop(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    
    """One epoch training"""
    
    # set modules as train()
    for module in module_list:
        module.train()

    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = []
    acc_vals = []

    for idx, data in enumerate(train_loader):

        input, target, index = data
        input = input.float()

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        # FORWARD
        
        feat_s, logit_s = model_s(input, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        loss_kd = 0
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc, acc5 = accuracy(logit_s, target, topk=(1))
        losses.append(loss.item())
        acc_vals.append(acc[0])

        # BACKWARD

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print info
        if idx % 5 == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_dataloader)}]\t'
                  f'Loss: {np.mean(losses)}\t'
                  f'Acc: {np.mean(acc_vals)}\t')

    print(f'Loss: {np.mean(losses)}\tAcc: {np.mean(acc_vals)}\t'))
    return np.mean(acc_vals), np.mean(losses)


def train():

    model_t = load_teacher(opt.path_t, n_cls, opt.dataset)
    model_s = create_model(opt.model_s, n_cls, opt.dataset)

    data = torch.randn(2, 3, 84, 84)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_kd = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    # classification loss
    criterion_list.append(criterion_cls)    
    # KL divergence loss
    criterion_list.append(criterion_div)   
    # other knowledge distillation loss
    criterion_list.append(criterion_kd)     

    # optimizer
    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)
    