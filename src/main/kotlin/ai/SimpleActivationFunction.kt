package ai

import java.lang.Math.pow
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.tanh

enum class SimpleActivationFunction : ActivationFunction {
    TANH {
        override fun invoke(t: Double) = tanh(t)

        override fun derivative(t: Double) = 1.0 - t * t // approx. of 1.0 - tanh(t)*tanh(t)
    },
    SIGMOID {
        override fun invoke(t: Double) = 1 / (1 + exp(-t))

        override fun derivative(t: Double) = this(t) * (1 - this(t))
    },
    SOFTSIGN {
        override fun invoke(t: Double) = t / (1 + abs(t))

        override fun derivative(t: Double) = 1 / pow((1 + abs(t)), 2.0)
    },
    ID {
        override fun invoke(t: Double) = t

        override fun derivative(t: Double) = 1.0
    },

    // More examples https://en.wikipedia.org/wiki/Activation_function
}