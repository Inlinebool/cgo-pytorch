import torch
from loguru import logger
from nlgeval import NLGEval
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def train_val_loss(model, train_dataset, val_dataset, num_workers, loss_fn,
                   params, model_save_path, save_every, device):
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
            clip_grad_norm_(model.parameters(), 0.25)
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


def train_val_meteor(model, train_dataset, val_dataset, val_cap_dataset,
                     word_map, reversed_word_map, num_workers, loss_fn, params,
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
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False)
    val_cap_dataloader = DataLoader(dataset=val_cap_dataset, shuffle=False)
    last_loss = 0.0

    # first prepare val refs
    logger.info('preparing references...')
    ref_img = {}
    for cap_label in val_dataloader:
        image_id = str(cap_label['image_id'][0])
        if image_id not in ref_img:
            ref_img[image_id] = []
        seq, seq_length = cap_label['label']
        ref_img[image_id].append(
            [reversed_word_map[x] for x in seq[0][1:seq_length[0] + 1]])

    nlg = NLGEval(False, True, True, ['Bleu_1', 'ROUGE_L', 'CIDEr'])
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
            clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            if i % 1 == 0:
                logger.info(
                    '[epoch: {0}/{1}, batch: {2}/{3}] loss: {4}'.format(
                        epoch + 1, params['epoch'], i + 1,
                        len(train_dataloader), running_loss))
                running_loss = 0.0
        with torch.no_grad():
            logger.info('validating epoch %d ... ' % (epoch + 1))
            model.eval()
            hyp = []
            ref = [[] for _ in range(5)]
            for val in tqdm(val_cap_dataloader):
                image_id = str(val['image_id'][0])
                image_features = [x.to(device) for x in val['inputs']][0]
                seq = torch.tensor([word_map['<start>']]).view(1,
                                                               -1).to(device)
                seq_length = torch.tensor([0]).view(1, -1).to(device)
                top_results = model.decode((image_features, seq, seq_length),
                                           word_map['<end>'],
                                           beam=1)
                decoded = [reversed_word_map[x] for x in top_results[0][1]]
                if decoded[-1] != '<end>':
                    logger.warning('decoded sentence not ending with <end>.')
                    logger.warning('image_id: {0}'.format(image_id))
                    logger.warning('decoded: {0}'.format(decoded))
                    hyp.append(' '.join(decoded))
                else:
                    hyp.append(' '.join(decoded[:-1]))
                for i in range(5):
                    ref[i].append(' '.join(ref_img[image_id][i]))

            logger.debug('sample 0 pred: {0}'.format(hyp[1234]))
            logger.debug('sample 0 ref 0: {0}'.format(ref[0][1234]))
            logger.debug('sample 0 ref 1: {0}'.format(ref[1][1234]))
            logger.debug('sample 0 ref 2: {0}'.format(ref[2][1234]))
            logger.debug('sample 0 ref 3: {0}'.format(ref[3][1234]))
            logger.debug('sample 0 ref 4: {0}'.format(ref[4][1234]))
            logger.debug('sample 1 pred: {0}'.format(hyp[2345]))
            logger.debug('sample 1 ref 0: {0}'.format(ref[0][2345]))
            logger.debug('sample 1 ref 1: {0}'.format(ref[1][2345]))
            logger.debug('sample 1 ref 2: {0}'.format(ref[2][2345]))
            logger.debug('sample 1 ref 3: {0}'.format(ref[3][2345]))
            logger.debug('sample 1 ref 4: {0}'.format(ref[4][2345]))
            metrics = nlg.compute_metrics(ref, hyp)
            meteor = metrics['METEOR']
            delta_loss = meteor - last_loss
            last_loss = meteor

        logger.info('val METEOR after epoch %d: %.10f' % (epoch + 1, meteor))
        logger.info('METEOR change after last epoch: %.10f' % delta_loss)
        scheduler.step()

        if not (epoch + 1) % save_every:
            with open('{0}_{1}.pkl'.format(model_save_path, epoch + 1),
                      'wb') as fp:
                logger.info("writing checkpoint " + str(epoch + 1))
                torch.save(model, fp)

    logger.info("training complete. writing final model.")
    with open('{0}.pkl'.format(model_save_path), 'wb') as fp:
        torch.save(model, fp)
