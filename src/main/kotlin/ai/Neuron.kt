package ai

import java.lang.Math.random

private const val LEARNING_RATE = 0.15
private const val EXPERIENCE_FACTOR = 0.5 // this is called MOMENTUM in literature

class Neuron(private val index: Int,
             private val activationFunction: ActivationFunction,
             outputCount: Int) {
    var output = 1.0
    private var gradient = 0.0

    val memory = Array(outputCount, init = { random() })
    private val experience = Array(outputCount, init = { 0.0 })


    fun think(previousLayer: Layer) {
        val t = previousLayer.sumByDouble { it.output * it.memory[index] }
        output = activationFunction(t)
    }

    fun learnFromResult(correctResult: Double) {
        val delta = correctResult - output
        gradient = delta * activationFunction.derivative(output)
    }

    fun learnFromNextLayer(nextLayer: Layer) {
        val delta = nextLayer.dropLast(1).zip(memory, { neuron, memory ->
            neuron.gradient * memory
        }).sumByDouble { it }
        gradient = delta * activationFunction.derivative(output)
    }

    fun remember(previousLayer: Layer) {
        previousLayer.forEach {
            val oldExperience = it.experience[index]
            val newExperience = LEARNING_RATE * it.output * gradient + EXPERIENCE_FACTOR * oldExperience
            it.memory[index] += newExperience
            it.experience[index] = newExperience
        }
    }

}


