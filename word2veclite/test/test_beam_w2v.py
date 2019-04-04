import argparse
import json
import logging
import os

import apache_beam as beam
import tensorflow as tf

from apache_beam.options.pipeline_options import PipelineOptions


def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


@singleton
class Model:
    def __init__(self, checkpoint):
        with tf.Graph().as_default() as graph:
            sess = tf.InteractiveSession()
            saver = tf.train.import_meta_graph(os.path.join(checkpoint, 'export.meta'))
            saver.restore(sess, os.path.join(checkpoint, 'export'))

            inputs = json.loads(tf.get_collection('inputs')[0])
            outputs = json.loads(tf.get_collection('outputs')[0])

            self.x = graph.get_tensor_by_name(inputs['image'])
            self.p = graph.get_tensor_by_name(outputs['scores'])
            self.input_key = graph.get_tensor_by_name(inputs['key'])
            self.output_key = graph.get_tensor_by_name(outputs['key'])
            self.sess = sess


class PredictDoFn(beam.DoFn):
    def process(self, element, checkpoint):
        model = Model(checkpoint)
        output_key, pred = model.sess.run(
            [model.output_key, model.p],
            feed_dict={model.input_key: element, model.x: element})


def run():
    with beam.Pipeline(options=PipelineOptions()) as p:
        lines = p | 'Create' >> beam.Create(['Exception: Index out of bounds',
                                             'ValueError: int cannot be typecasted to str',
                                             'KeyError: Key 1 does not exist in dictionary'])
    images = (p | 'ReadFromText' >> beam.io.ReadFromText(known_args.input)
              | 'ConvertToDict'>> beam.Map(_to_dictionary))
    predictions = images | 'Prediction' >> beam.ParDo(Word2VecFn(), known_args.model)
    predictions | 'WriteToText' >> beam.io.WriteToText(known_args.output)
    logging.getLogger().setLevel(logging.INFO)
    p.run()


if __name__ == '__main__':
    run()
