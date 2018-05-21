package ai

import java.lang.Math.random

private const val NEW_KNOWLEDGE_FACTOR = 0.15
private const val OLD_KNOWLEDGE_FACTOR = 0.5

class Neuron(private val index: Int,
             private val activationFunction: ActivationFunction,
             outputCount: Int) {
    var output = 1.0
    var gradient = 0.0

    val memory = DoubleArray(outputCount, init = { random() })
    val experience = DoubleArray(outputCount, init = { 0.0 })


    fun think(previousLayer: Layer) {
        val t = (0 until previousLayer.size).sumByDouble { previousLayer[it].output * previousLayer[it].memory[index] }
        output = activationFunction.compute(t)
    }

    fun learnFromResult(correctResult: Double) {
        val delta = correctResult - output
        gradient = delta * activationFunction.derivative(output)
    }

    fun learnFromNextLayer(nextLayer: Layer) {
        val delta = (0 until nextLayer.size - 1).sumByDouble { memory[it] * nextLayer[it].gradient }
        gradient = delta * activationFunction.derivative(output)
    }

    fun remember(previousLayer: Layer) {
        previousLayer.forEach {
            val oldExperience = it.experience[index]
            val newExperience = NEW_KNOWLEDGE_FACTOR * it.output * gradient + OLD_KNOWLEDGE_FACTOR * oldExperience
            it.memory[index] = it.memory[index] + newExperience
            it.experience[index] = newExperience
        }
    }

}


//In summary: when performing gradient descent, learning rate measures how much the current situation
// affects the next step, while momentum measures how much past steps affect the next step.