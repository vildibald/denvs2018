package ai

import kotlin.math.sqrt


typealias Layer = Array<Neuron>
typealias Network = Array<Layer>

private const val SMOOTHING_FACTOR = 100.0

class Brain(activationFunction: ActivationFunction, topology: Array<Int>) {
    private var error = 0.0
    var recentAverageError = 0.0
    private val network: Network

    private val lambdaW = 0.001


    init {
        val layerCount = topology.size
        network = Network(size = layerCount, init = {
            // it znamena i-tu vrstvu
            val neuronCount = topology[it] + 1
            val outputCount = if (it == layerCount - 1) 0 else topology[it + 1]
            Layer(size = neuronCount, init = {
                Neuron(it, activationFunction, outputCount)
            })
        })
    }

    fun think(inputs: Array<Double>) {
        (0 until inputs.size).forEach { network[0][it].output = inputs[it] }

        (1 until network.size).forEach { i ->
            val previousLayer = network[i - 1]
            val currentLayer = network[i]
            currentLayer.dropLast(1).forEach { it.think(previousLayer) }
        }
    }

    fun learn(correctResults: Array<Double>) {
        error = computeError(correctResults)
        recentAverageError = computeRecentAverageError()
        teachOutputLayer(correctResults)
        teachLayers()
        remember()
    }

    private fun computeError(result: Array<Double>): Double {
        val outputLayer = network.last()
        var error = (0 until outputLayer.lastIndex).map { result[it] - outputLayer[it].output }
                .sumByDouble { it * it }
        error /= outputLayer.lastIndex
        error = sqrt(error)
        error += lambdaW * sqrt(network.sumByDouble {
            it.sumByDouble { it.memory.sumByDouble { it * it } }
        })
        return error
    }

    private fun computeRecentAverageError() = (recentAverageError * SMOOTHING_FACTOR
            + error) / (SMOOTHING_FACTOR + 1.0)


    private fun teachOutputLayer(result: Array<Double>) {
        network.last().dropLast(1).zip(result).forEach {
            (neuron, correctResult) -> neuron.learnFromResult(correctResult)
        }
    }

    private fun teachLayers() {
        (network.lastIndex - 1 downTo 1).forEach {
            val layer = network[it]
            val nextLayer = network[it + 1]
            layer.forEach { it.learnFromNextLayer(nextLayer) }
        }
    }

    private fun remember() {
        (network.lastIndex downTo 1).forEach {
            val layer = network[it]
            val previousLayer = network[it - 1]
            layer.dropLast(1).forEach{ it.remember(previousLayer)}
        }
    }

    fun results() = network.last().map { it.output }.dropLast(1).toTypedArray()

}