import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader


def train(model, train_dataset, val_dataset, num_workers, loss_fn, params,
          model_save_path, save_every, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=params['decay_every'],
        gamma=params['decay_rate'])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=params['batch_size'],
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=params['batch_size'],
                                num_workers=0,
                                pin_memory=True,
                                shuffle=True)
    last_loss = 0.0
    for epoch in range(params['epoch']):
        running_loss = 0.0
        logger.info('training epoch %d ... ' % (epoch + 1))
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = [x.to(device) for x in batch['inputs']]
            label = [x.to(device) for x in batch['label']]
            prediction = model(inputs)
            loss = loss_fn(prediction, label)
            running_loss += loss

            loss.backward()
            optimizer.step()

            if i % 1 == 0:
                logger.info(
                    '[epoch: {0}/{1}, batch: {2}/{3}] loss: {4}'.format(
                        epoch + 1, params['epoch'], i + 1,
                        len(train_dataloader), running_loss))
                running_loss = 0.0
        with torch.no_grad():
            logger.info('validating epoch %d ... ' % (epoch + 1))
            running_loss = 0.0
            model.eval()
            for i, batch in enumerate(val_dataloader):
                inputs = [x.to(device) for x in batch['inputs']]
                label = [x.to(device) for x in batch['label']]
                prediction = model(inputs)
                loss = loss_fn(prediction, label)
                running_loss += loss
            running_loss /= len(val_dataloader)
            delta_loss = running_loss - last_loss
            last_loss = running_loss

        logger.info('loss after epoch %d: %.10f' % (epoch + 1, running_loss))
        logger.info('loss change after last epoch: %.10f' % delta_loss)
        scheduler.step()

        if not (epoch + 1) % save_every:
            with open('{0}_{1}.pkl'.format(model_save_path, epoch + 1),
                      'wb') as fp:
                logger.info("writing checkpoint " + str(epoch + 1))
                torch.save(model, fp)

    logger.info("training complete. writing final model.")
    with open('{0}.pkl'.format(model_save_path), 'wb') as fp:
        torch.save(model, fp)
