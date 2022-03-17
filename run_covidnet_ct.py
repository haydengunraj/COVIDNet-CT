"""
Training/testing/inference script for COVID-Net CT models for COVID-19 detection in CT images.
"""

import os
import sys
import cv2
import json
import shutil
import numpy as np
from math import ceil
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import COVIDxCTDataset
from data_utils import auto_body_crop
from utils import parse_args

# Dict keys
TRAIN_OP_KEY = 'train_op'
TF_SUMMARY_KEY = 'tf_summaries'
LOSS_KEY = 'loss'

# Tensor names
IMAGE_INPUT_TENSOR = 'Placeholder:0'
LABEL_INPUT_TENSOR = 'Placeholder_1:0'
CLASS_PRED_TENSOR = 'ArgMax:0'
CLASS_PROB_TENSOR = 'softmax_tensor:0'
TRAINING_PH_TENSOR = 'is_training:0'
LOSS_TENSOR = 'add:0'

# Names for train checkpoints
CKPT_NAME = 'model.ckpt'
MODEL_NAME = 'COVID-Net_CT'

# Output directory for storing runs
OUTPUT_DIR = 'output'

# Class names ordered by class index
CLASS_NAMES = ('Normal', 'Pneumonia', 'COVID-19')


def dense_grad_filter(gvs):
    """Filter to apply gradient updates to dense layers only"""
    return [(g, v) for g, v in gvs if 'dense' in v.name]


def simple_summary(tag_to_value, tag_prefix=''):
    """Summary object for a dict of python scalars"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag_prefix + tag, simple_value=value)
                             for tag, value in tag_to_value.items() if isinstance(value, (int, float))])


def create_session():
    """Helper function for session creation"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def load_graph(meta_file):
    """Creates new graph and session"""
    graph = tf.Graph()
    with graph.as_default():
        # Create session and load model
        sess = create_session()

        # Load meta file
        print('Loading meta graph from ' + meta_file)
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    return graph, sess, saver


def load_ckpt(ckpt, sess, saver):
    """Helper for loading weights"""
    # Load weights
    if ckpt is not None:
        print('Loading weights from ' + ckpt)
        saver.restore(sess, ckpt)


def get_lr_scheduler(init_lr, global_step=None, decay_steps=None, schedule_type='cosine'):
    """Helper for making a learning rate scheduler"""
    if schedule_type == 'constant':
        return init_lr
    elif schedule_type == 'cosine':
        return tf.train.cosine_decay(init_lr, global_step, decay_steps, alpha=0.01)
    elif schedule_type == 'exp':
        return tf.train.exponential_decay(init_lr, global_step, decay_steps)


class Metrics:
    """Lightweight class for tracking metrics"""
    def __init__(self):
        num_classes = len(CLASS_NAMES)
        self.labels = list(range(num_classes))
        self.class_names = CLASS_NAMES
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)

    def update(self, y_true, y_pred):
        self.confusion_matrix = self.confusion_matrix + confusion_matrix(y_true, y_pred, labels=self.labels)

    def reset(self):
        self.confusion_matrix *= 0

    def values(self):
        conf_matrix = self.confusion_matrix.astype('float')
        metrics = {
            'accuracy': np.diag(conf_matrix).sum() / conf_matrix.sum(),
            'confusion matrix': self.confusion_matrix.copy()
        }
        sensitivity = np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=1), 1)
        pos_pred_val = np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=0), 1)
        for cls, idx in zip(self.class_names, self.labels):
            metrics['{} {}'.format(cls, 'sensitivity')] = sensitivity[idx]
            metrics['{} {}'.format(cls, 'PPV')] = pos_pred_val[idx]
        return metrics


class COVIDNetCTRunner:
    """Primary training/testing/inference class"""
    def __init__(self, meta_file, ckpt=None, data_dir=None, input_height=512, input_width=512, max_bbox_jitter=0.025,
                 max_rotation=10, max_shear=0.15, max_pixel_shift=10, max_pixel_scale_change=0.2):
        self.meta_file = meta_file
        self.ckpt = ckpt
        self.input_height = input_height
        self.input_width = input_width
        if data_dir is None:
            self.dataset = None
        else:
            self.dataset = COVIDxCTDataset(
                data_dir,
                image_height=input_height,
                image_width=input_width,
                max_bbox_jitter=max_bbox_jitter,
                max_rotation=max_rotation,
                max_shear=max_shear,
                max_pixel_shift=max_pixel_shift,
                max_pixel_scale_change=max_pixel_scale_change
            )

    def load_graph(self):
        """Creates new graph and session"""
        graph = tf.Graph()
        with graph.as_default():
            # Create session and load model
            sess = create_session()

            # Load meta file
            print('Loading meta graph from ' + self.meta_file)
            saver = tf.train.import_meta_graph(self.meta_file, clear_devices=True)
        return graph, sess, saver

    def load_ckpt(self, sess, saver):
        """Helper for loading weights"""
        # Load weights
        if self.ckpt is not None:
            print('Loading weights from ' + self.ckpt)
            saver.restore(sess, self.ckpt)

    def trainval(self, epochs, output_dir, batch_size=1, learning_rate=0.001, momentum=0.9,
                 fc_only=False, train_split_file='train.txt', val_split_file='val.txt',
                 log_interval=20, val_interval=1000, save_interval=1000):
        """Run training with intermittent validation"""
        ckpt_path = os.path.join(output_dir, CKPT_NAME)
        graph, sess, saver = self.load_graph()
        with graph.as_default():
            # Save graph without optimizer
            tf.train.export_meta_graph(filename=os.path.join(output_dir, 'model.meta'))

            # Create train dataset
            dataset, num_images, batch_size = self.dataset.train_dataset(train_split_file, batch_size, True)
            data_next = dataset.make_one_shot_iterator().get_next()
            num_iters = ceil(num_images/batch_size)*epochs

            # Create optimizer
            global_step = tf.train.get_or_create_global_step()
            scheduler = get_lr_scheduler(
                learning_rate, schedule_type='cosine_decay',
                global_step=global_step, decay_steps=num_iters, alpha=0.01)
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=scheduler,
                momentum=momentum
            )
            optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

            # Create train op
            loss = graph.get_tensor_by_name(LOSS_TENSOR)
            grad_vars = optimizer.compute_gradients(loss)
            if fc_only:
                grad_vars = dense_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)

            # Load checkpoint
            sess.run(tf.global_variables_initializer())
            self.load_ckpt(sess, saver)

            # Create feed and fetch dicts
            feed_dict = {TRAINING_PH_TENSOR: True}
            fetch_dict = {
                TRAIN_OP_KEY: train_op,
                LOSS_KEY: LOSS_TENSOR
            }

            # Add summaries
            summary_writer = tf.summary.FileWriter(os.path.join(output_dir, 'events'), graph)
            fetch_dict[TF_SUMMARY_KEY] = self._get_train_summary_op(graph)

            # Create validation function
            run_validation = self._get_validation_fn(sess, batch_size, val_split_file)

            print('Starting baseline validation')
            metrics = run_validation()
            self._log_and_print_metrics(metrics, 0, summary_writer)

            # Training loop
            print('Training with batch_size {} for {} steps'.format(batch_size, num_iters))
            loss_accum = 0
            for i in range(num_iters):
                # Run training step
                data = sess.run(data_next)
                feed_dict[IMAGE_INPUT_TENSOR] = data['image']
                feed_dict[LABEL_INPUT_TENSOR] = data['label']
                results = sess.run(fetch_dict, feed_dict)

                # Log and save
                step = i + 1
                if step % log_interval == 0:
                    summary_writer.add_summary(results[TF_SUMMARY_KEY], step)
                    print('[step: {}, loss: {}]'.format(step, loss_accum/log_interval))
                    loss_accum = 0
                if step%save_interval == 0:
                    print('Saving checkpoint at step {}'.format(step))
                    saver.save(sess, ckpt_path, global_step=step, write_meta_graph=False)
                if val_interval > 0 and step % val_interval == 0:
                    print('Starting validation at step {}'.format(step))
                    metrics = run_validation()
                    self._log_and_print_metrics(metrics, step, summary_writer)

            print('Saving checkpoint at last step')
            saver.save(sess, ckpt_path, global_step=num_iters, write_meta_graph=False)

    def test(self, batch_size=1, test_split_file='test.txt', plot_confusion=False):
        """Run test on a checkpoint"""
        graph, sess, saver = self.load_graph()
        with graph.as_default():
            # Load checkpoint
            self.load_ckpt(sess, saver)

            # Run test
            print('Starting test')
            metrics = self._get_validation_fn(sess, batch_size, test_split_file)()
            self._log_and_print_metrics(metrics)

            if plot_confusion:
                # Plot confusion matrix
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion matrix'],
                                              display_labels=CLASS_NAMES)
                disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='horizontal', values_format='.5g')
                plt.show()

    def infer(self, image_file, autocrop=False):
        """Run inference on the given image"""
        # Load and preprocess image
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if autocrop:
            image, _ = auto_body_crop(image)
        image = cv2.resize(image, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        image = image.astype(np.float32)/255.0
        image = np.expand_dims(np.stack((image, image, image), axis=-1), axis=0)

        # Create feed dict
        feed_dict = {IMAGE_INPUT_TENSOR: image, TRAINING_PH_TENSOR: False}

        # Run inference
        graph, sess, saver = self.load_graph()
        with graph.as_default():
            # Load checkpoint
            self.load_ckpt(sess, saver)

            # Add training placeholder if present
            try:
                sess.graph.get_tensor_by_name(TRAINING_PH_TENSOR)
                feed_dict[TRAINING_PH_TENSOR] = False
            except KeyError:
                pass

            # Run image through model
            class_, probs = sess.run([CLASS_PRED_TENSOR, CLASS_PROB_TENSOR], feed_dict=feed_dict)
            print('\nPredicted Class: ' + CLASS_NAMES[class_[0]])
            print('Confidences: ' + ', '.join(
                '{}: {}'.format(name, conf) for name, conf in zip(CLASS_NAMES, probs[0])))
            print('**DISCLAIMER**')
            print('Do not use this prediction for self-diagnosis. '
                  'You should check with your local authorities for '
                  'the latest advice on seeking medical assistance.')

    def _get_validation_fn(self, sess, batch_size=1, val_split_file='val.txt'):
        """Creates validation function to call in self.trainval() or self.test()"""
        # Create val dataset
        dataset, num_images, batch_size = self.dataset.validation_dataset(val_split_file, batch_size)
        dataset = dataset.repeat()  # repeat so there is no need to reconstruct it
        data_next = dataset.make_one_shot_iterator().get_next()
        num_iters = ceil(num_images / batch_size)

        # Create running accuracy metric
        metrics = Metrics()

        # Create feed and fetch dicts
        fetch_dict = {'classes': CLASS_PRED_TENSOR}
        feed_dict = {}

        # Add training placeholder if present
        try:
            sess.graph.get_tensor_by_name(TRAINING_PH_TENSOR)
            feed_dict[TRAINING_PH_TENSOR] = False
        except KeyError:
            pass

        def run_validation():
            metrics.reset()
            for i in range(num_iters):
                data = sess.run(data_next)
                feed_dict[IMAGE_INPUT_TENSOR] = data['image']
                results = sess.run(fetch_dict, feed_dict)
                metrics.update(data['label'], results['classes'])
            return metrics.values()

        return run_validation

    @staticmethod
    def _log_and_print_metrics(metrics, step=None, summary_writer=None, tag_prefix='val/'):
        """Helper for logging and printing"""
        # Pop temporarily and print
        cm = metrics.pop('confusion matrix')
        print('\tconfusion matrix:')
        print('\t' + str(cm).replace('\n', '\n\t'))

        # Print scalar metrics
        for name, val in sorted(metrics.items()):
            print('\t{}: {}'.format(name, val))

        # Log scalar metrics
        if summary_writer is not None:
            summary = simple_summary(metrics, tag_prefix)
            summary_writer.add_summary(summary, step)

        # Restore confusion matrix
        metrics['confusion matrix'] = cm

    @staticmethod
    def _get_train_summary_op(graph, tag_prefix='train/'):
        loss = graph.get_tensor_by_name(LOSS_TENSOR)
        loss_summary = tf.summary.scalar(tag_prefix + 'loss', loss)
        return loss_summary


if __name__ == '__main__':
    # Suppress most TF messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    mode, args = parse_args(sys.argv[1:])

    # Create full paths
    meta_file = os.path.join(args.model_dir, args.meta_name)
    ckpt = os.path.join(args.model_dir, args.ckpt_name)

    # Create runner
    if mode == 'train':
        augmentation_kwargs = dict(
            max_bbox_jitter=args.max_bbox_jitter,
            max_rotation=args.max_rotation,
            max_shear=args.max_shear,
            max_pixel_shift=args.max_pixel_shift,
            max_pixel_scale_change=args.max_pixel_scale_change
        )
    else:
        augmentation_kwargs = {}
    runner = COVIDNetCTRunner(
        meta_file,
        ckpt=ckpt,
        data_dir=args.data_dir,
        input_height=args.input_height,
        input_width=args.input_width,
        **augmentation_kwargs
    )

    if mode == 'train':
        # Create output_dir and save run settings
        output_dir = os.path.join(OUTPUT_DIR, MODEL_NAME + args.output_suffix)
        os.makedirs(output_dir, exist_ok=False)
        with open(os.path.join(output_dir, 'run_settings.json'), 'w') as f:
            json.dump(vars(args), f)

            # Run trainval
            runner.trainval(
                args.epochs,
                output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                fc_only=args.fc_only,
                train_split_file=args.train_split_file,
                val_split_file=args.val_split_file,
                log_interval=args.log_interval,
                val_interval=args.val_interval,
                save_interval=args.save_interval
            )
    elif mode == 'test':
        # Run validation
        runner.test(
            batch_size=args.batch_size,
            test_split_file=args.test_split_file,
            plot_confusion=args.plot_confusion
        )
    elif mode == 'infer':
        # Run inference
        runner.infer(args.image_file, not args.no_crop)
