package ai

import kotlin.math.tanh

class ActivationFunction {
    fun compute(t: Double) = tanh(t)

    fun derivative(t: Double) = 1.0 - t * t // approximation of 1.0 - tanh(s)*tanh(s)
}
