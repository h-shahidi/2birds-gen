from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import nn_ops


class Decoder:
    def __init__(self, flags, vocab, reward, is_training):
        self.flags = flags
        self.vocab = vocab
        self.reward = reward
        self.cell = tf.contrib.rnn.LSTMCell(
                    self.flags.decoder_hidden_size,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    state_is_tuple=True)

        with tf.variable_scope("embedding"), tf.device('/cpu:0'):
            self.embedding = tf.get_variable(
                'word_embedding', trainable=(self.flags.fix_word_vec==False),
                initializer=tf.constant(self.vocab.word_vecs), dtype=tf.float32)

    def attention(self, decoder_state, encoder_states, encoder_features, passage_mask, coverage, v, w_c):
        '''
        decoder_state: Tuple of [batch_size, decoder_hidden_size]
        encoder_states: [batch_size, passage_len, encoder_dim]
        encoder_features: [batch_size,passage_len,attention_vec_size]
        passage_mask: [batch_size, passage_len]
        v: [1,1, attention_vec_size]
        w_c: [1,1, attention_vec_size]
        coverage: [batch_size, passage_len]
        '''
        with tf.variable_scope("Attention"):
            state_features = linear(decoder_state, self.flags.attention_vec_size, True) # [batch_size, attention_vec_size]
            state_features = tf.expand_dims(state_features, 1) # [batch_size, 1, attention_vec_size]
            all_features = encoder_features + state_features # [batch_size,passage_len,attention_vec_size]
            if self.flags.use_coverage and coverage is not None:
                coverage_features = tf.expand_dims(coverage, axis=-1) * w_c # [batch_size, passage_len, attention_vec_size]
                all_features += coverage_features
            e = tf.reduce_sum(v * tf.tanh(all_features), axis=-1) # [batch_size, passage_len]
            attn_dist = nn_ops.softmax(e) # [batch_size, passage_len]
            attn_dist *= passage_mask

            if coverage is not None: # Update coverage vector
                coverage += attn_dist
            else: # first step of training
                coverage = attn_dist

            # Calculate the context vector from attn_dist and encoder_states
            # shape (batch_size, attn_size).
            context_vector = tf.reduce_sum(tf.expand_dims(attn_dist, axis=-1) * encoder_states, axis=1) # [batch_size, encoder_dim]
        return context_vector, attn_dist, coverage

    def embedding_lookup(self, inputs):
        '''
        inputs: list of [batch_size], int32
        '''
        if type(inputs) is list:
            return [tf.nn.embedding_lookup(self.embedding, x) for x in inputs]
        else:
            return tf.nn.embedding_lookup(self.embedding, inputs)

    def step(self, state_t_1, context_t_1, coverage_t_1, word_t, encoder_states, encoder_features,
             passage_words, passage_mask, v, w_c):
        '''
        state_t_1: Tuple of [batch_size, decoder_hidden_size]
        context_t_1: [batch_size, encoder_dim]
        coverage_t_1: [batch_size, passage_len]
        word_t: [batch_size, word_dim]
        encoder_states: [batch_size, passage_len, encoder_dim]
        encoder_features: [batch_size,attn_length,attention_vec_size]
        passage_mask: [batch_size, passage_len]
        v: [1,1, attention_vec_size]
        w_c: [1,1, attention_vec_size]
        '''
        cell_input = linear([word_t, context_t_1], self.flags.attention_vec_size, True)

        # Run the decoder RNN cell. 
        cell_output, state_t = self.cell(cell_input, state_t_1)

        context_t, attn_dist_t, coverage_t = self.attention(state_t, encoder_states, encoder_features, 
                                                          passage_mask, coverage_t_1, v, w_c)

        pointer_gen_t = None
        if self.flags.pointer_gen:
            with tf.variable_scope('pointer_gen'):
                pointer_gen_t = linear([context_t, state_t.c, state_t.h, cell_input], 1, True) # [batch_size, 1]
                pointer_gen_t = tf.sigmoid(pointer_gen_t)

        # Concatenate the cell_output and the context vector, and pass them through a linear layer.
        with tf.variable_scope("attention_output_projection"):
            output_t = linear([cell_output] + [context_t], self.flags.decoder_hidden_size, True)

        with tf.variable_scope('output_projection'):
            w = tf.get_variable('w', [self.flags.decoder_hidden_size, self.vocab.vocab_size+1], dtype=tf.float32)
            b = tf.get_variable('b', [self.vocab.vocab_size +1], dtype=tf.float32)
            vocab_score_t = tf.nn.xw_plus_b(output_t, w, b)
            vocab_score_t = tf.nn.softmax(vocab_score_t)

            # For pointer-generator model, calculate the final distribution.
            if self.flags.pointer_gen:
                vocab_score_t = self.calculate_final_dist(vocab_score_t, attn_dist_t, pointer_gen_t, passage_words, passage_mask)
            vocab_score_t = _clip_and_normalize(vocab_score_t, 1e-6)

        return (state_t, context_t, coverage_t, attn_dist_t, pointer_gen_t, vocab_score_t)

    def train(self, encoder_dim, encoder_states, encoder_features, passage_words, passage_mask,
              init_state, decoder_inputs, question_words, loss_weights, mode='ce_train'):
        '''
        encoder_dim: int value
        encoder_states: [batch_size, passage_len, encoder_dim].
        passage_words: [batch_size, passage_len] int32
        passage_mask: [batch_size, passage_len] 0/1
        init_state: Tuple of [batch_size, decoder_hidden_size]
        decoder_inputs: [batch_size, max_dec_steps].
        question_words: [batch_size, max_dec_steps]
        '''
        batch_size = tf.shape(encoder_states)[0]

        decoder_inputs = tf.unstack(decoder_inputs, axis=1) # max_enc_steps * [batch_size]
        question_words_unstack = tf.unstack(question_words, axis=1)

        # initialize all the variables
        state_t_1 = init_state
        context_t_1 = tf.zeros([batch_size, encoder_dim])
        coverage_t_1 = None

        # store variables from each time step
        coverages = []
        attn_dists = []
        pointer_gens = []
        vocab_scores = []
        output_words = []
        with tf.variable_scope("attention_decoder"):
            # Get the weight vectors v and W_c (W_c is for coverage)
            v = tf.get_variable("v", [self.flags.attention_vec_size])
            v = tf.expand_dims(tf.expand_dims(v, axis=0), axis=0)
            w_c = None
            if self.flags.use_coverage:
                with tf.variable_scope("coverage"):
                    w_c = tf.get_variable("w_c", [self.flags.attention_vec_size])
                    w_c = tf.expand_dims(tf.expand_dims(w_c, axis=0), axis=0)

            word_idx_t = decoder_inputs[0] # [batch_size]
            for i in range(self.flags.max_question_len):
                if mode in ('ce_train', 'loss'): 
                    word_idx_t = decoder_inputs[i]
                word_t = self.embedding_lookup(word_idx_t)

                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                (state_t, context_t, coverage_t, attn_dist_t, pointer_gen_t, output_t) = self.step(
                    state_t_1, context_t_1, coverage_t_1, word_t, encoder_states, 
                    encoder_features, passage_words, passage_mask, v, w_c)

                coverages.append(coverage_t)
                attn_dists.append(attn_dist_t)
                pointer_gens.append(pointer_gen_t)
                vocab_scores.append(output_t)

                state_t_1 = state_t
                context_t_1 = context_t
                coverage_t_1 = coverage_t

                if mode == 'greedy':
                    word_idx_t = tf.argmax(output_t, 1) # [batch_size]
                    word_idx_t = tf.reshape(word_idx_t, [-1]) # [batch_size]
                elif mode == 'sample':
                    log_score_t = tf.log(output_t) # [batch_size, vsize]
                    word_idx_t = tf.multinomial(log_score_t, 1) # [batch_size, 1]
                    word_idx_t = tf.reshape(word_idx_t, [-1]) # [batch_size]
                elif mode in ('ce_train', 'loss'):
                    word_idx_t = question_words_unstack[i]
                else:
                    raise ValueError('unknown generating mode %s' % mode)
                output_words.append(word_idx_t)

        if len(output_words)!=0:
            output_words = tf.stack(output_words, axis=1) # [batch_size, max_dec_steps]

        vocab_scores = tf.stack(vocab_scores, axis=1) # [batch_size, max_dec_steps, vocab]

        # calculating loss
        loss = None
        if mode in ('ce_train', 'loss'):
            batch_loss = ce_loss(vocab_scores, question_words, loss_weights) # [batch_size]
            if mode == 'loss': 
                batch_loss *= self.reward # multiply with rewards
            loss = tf.reduce_mean(batch_loss)
            # Calculate coverage loss from the attention distributions
            if self.flags.use_coverage:
                with tf.variable_scope('coverage_loss'):
                    cov_loss = coverage_loss(attn_dists, loss_weights)
                loss = loss + self.flags.coverage_loss_const * cov_loss

        # accuracy is calculated only under 'ce_train'
        if mode == 'ce_train':
            accuracy = _mask_and_get_accuracy(vocab_scores, question_words, loss_weights)
            return accuracy, loss, output_words
        else:
            return None, loss, output_words

    def decode(self, state_t_1, context_t_1, coverage_t_1, word_t,
               encoder_states, encoder_features, passage_words, passage_mask):
        with tf.variable_scope("attention_decoder"):
            v = tf.get_variable("v", [self.flags.attention_vec_size])
            v = tf.expand_dims(tf.expand_dims(v, axis=0), axis=0)
            w_c = None
            if self.flags.use_coverage:
                with tf.variable_scope("coverage"):
                    w_c = tf.get_variable("w_c", [self.flags.attention_vec_size])
                    w_c = tf.expand_dims(tf.expand_dims(w_c, axis=0), axis=0)

            word_t_representation = self.embedding_lookup(word_t)

            (state_t, context_t, coverage_t, attn_dist_t, p_gen_t, output_t) = self.step(
                state_t_1, context_t_1, coverage_t_1, word_t_representation, encoder_states, 
                encoder_features, passage_words, passage_mask, v, w_c)

            vocab_scores = tf.log(output_t)
            greedy_prediction = tf.reshape(tf.argmax(output_t, 1),[-1]) # calcualte greedy
            multinomial_prediction = tf.reshape(tf.multinomial(vocab_scores, 1),[-1]) # calculate multinomial
            topk_log_probs, topk_ids = tf.nn.top_k(vocab_scores, self.flags.beam_size) # calculate topK

        return (state_t, context_t, coverage_t, attn_dist_t, p_gen_t, output_t, topk_log_probs, topk_ids,
                greedy_prediction, multinomial_prediction)

    def calculate_final_dist(self, vocab_dist, attn_dist, pointer_gen, passage_words, passage_mask=None):
        '''
        vocab_dist: [batch_size, vsize]
        attn_dist: [batch_size, passage_length]
        pointer_gen: [batch_size, 1]
        passage_words: [batch_size, passage_length]
        passage_mask: [batch_size, passage_length]
        '''
        input_shape = tf.shape(vocab_dist)
        batch_size = input_shape[0]
        vsize = input_shape[1]
        passage_length = tf.shape(passage_words)[1]

        with tf.variable_scope('final_distribution'):
            vocab_dist = pointer_gen * vocab_dist
            attn_dist = (1.0 - pointer_gen) * attn_dist

            # match attn_dist[batch_size, passage_length] to sparse one-hot representation [batch_size, passage_length, vsize]
            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, axis=1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, passage_length]) # shape (batch_size, passage_length)
            step_nums = tf.range(0, limit=passage_length) # [passage_length]
            step_nums = tf.expand_dims(step_nums, axis=0) # shape (1, passage_length)
            step_nums = tf.tile(step_nums, [batch_size, 1]) # shape (batch_size, passage_length)
            indices = tf.stack((batch_nums, step_nums, passage_words), axis=2) # shape (batch_size, passage_length, 3)
            indices = tf.reshape(indices, [-1, 3]) #[batch_size * passage_length, 3]
            indices = tf.cast(indices, tf.int64)

            shape = [batch_size, passage_length, vsize]
            shape = tf.cast(shape, tf.int64)

            attn_dist = tf.reshape(attn_dist, shape=[-1]) # [batch_size*passage_length]
            one_hot_spare_representation = tf.SparseTensor(indices=indices, values=attn_dist, dense_shape=shape) # [batch_size, passage_length, vsize]

            if passage_mask is not None:
                passage_mask = tf.expand_dims(passage_mask, axis=-1)
                one_hot_spare_representation = one_hot_spare_representation * passage_mask

            one_hot_spare_representation = tf.sparse_reduce_sum(one_hot_spare_representation, axis=1) # [batch_size, vsize]

            vocab_dist = tf.add(vocab_dist, one_hot_spare_representation)

        return vocab_dist # [batch_size, vsize]

def linear(args, output_size, bias=True, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(values=args, axis=1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return res + bias_term


def ce_loss(word_probs, targets, loss_weights):
    '''
    word_probs: [batch_size, max_dec_steps, vocab]
    targets: [batch_size, max_dec_steps]
    loss_weigts: [batch_size, max_dec_steps]
    '''
    input_shape = tf.shape(word_probs)
    vsize = input_shape[2]

    epsilon = 1.0e-6
    word_probs = _clip_and_normalize(word_probs, epsilon)

    one_hot_representation = tf.one_hot(targets, vsize)

    losses = -tf.reduce_sum(one_hot_representation * tf.log(word_probs), axis=-1) # [batch_size, max_dec_steps]

    if loss_weights != None:
        losses = losses * loss_weights
    batch_loss = tf.reduce_sum(losses, axis=-1)

    return batch_loss #[batch_size]


def coverage_loss(attn_dists, loss_weights):
    """
    Calculates the coverage loss from the attention distributions.

    Args:
        attn_dists: The attention distributions for each decoder timestep.
                    A list with length max_dec_steps containing tensors with shape (batch_size, attn_length)
        loss_weights: shape (batch_size, max_dec_steps).

    Returns:
        coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    coverage_losses = [] # Coverage loss per decoder time step. It will be a list with length max_dec_steps containing tensors with shape (batch_size).
    for dist in attn_dists:
        cov_loss = tf.reduce_sum(tf.minimum(dist, coverage), [1]) # calculate the coverage loss for this step
        coverage_losses.append(cov_loss)
        coverage += dist # update the coverage vector
    coverage_loss = _mask_and_avg(coverage_losses, loss_weights)
    return coverage_loss


def _mask_and_get_accuracy(values, targets, loss_weights):
    # values: [batch_size, step_size, vocab_size]
    # targets: [batch_size, step_size]
    values = tf.argmax(values,axis=2)
    x = tf.cast(values, dtype=tf.int32)
    y = tf.cast(targets, dtype=tf.int32)
    res = tf.equal(x, y)
    res = tf.cast(res, dtype=tf.float32)
    res = tf.multiply(res, loss_weights)
    return tf.reduce_sum(res)


def _mask_and_avg(values, loss_weights):
    """
    Applies mask to values then returns overall average (a scalar)

      Args:
        values: a list with length max_dec_steps containing arrays with shape (batch_size).
        loss_weights: tensor with shape (batch_size, max_dec_steps) containing 1s and 0s.

      Returns:
        a scalar
    """
    if loss_weights == None:
        return tf.reduce_mean(tf.stack(values, axis=0))

    dec_lens = tf.reduce_sum(loss_weights, axis=1) # shape (batch_size)
    values_per_step = [v * loss_weights[:,dec_step] for dec_step,v in enumerate(values)]
    result = sum(values_per_step)/dec_lens # shape (batch_size)
    return tf.reduce_mean(result) # overall average


def _clip_and_normalize(word_probs, epsilon):
    word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True)