package ai

import kotlin.math.sqrt


typealias Layer = Array<Neuron>
typealias Network = Array<Layer>

class Brain(activationFunction: ActivationFunction, topology: IntArray) {
    var error = 0.0
    var recentAverageError = 0.0
    private val network: Network

    private val lambdaW = 0.001
    private val recentAverageSmoothingFactor = 100.0

    init {
        val layerCount = topology.size
        network = Network(layerCount, init = {
            val neuronCount = topology[it] + 1
            val outputCount = if (it == layerCount - 1) 0 else topology[it + 1]
            Layer(neuronCount, init = {
                Neuron(it, activationFunction, outputCount)
            })
        })
    }

    fun think(inputs: DoubleArray) {
        for (i in 0 until inputs.size) {
            network[0][i].output = inputs[i]
        }

        for (i in 1 until network.size) {
            val previousLayer = network[i - 1]
            val currentLayer = network[i]
            for (j in 0 until currentLayer.lastIndex) {
                currentLayer[j].think(previousLayer)
            }
        }
    }

    fun learn(result: DoubleArray) {
        error = computeError(result)
        recentAverageError = computeRecentAverageError()
        teachOutputLayer(result)
        teachLayers()
        remember()
    }

    private fun computeError(result: DoubleArray): Double {
        val outputLayer = network.last()
        var error = 0.0
        for (i in 0 until outputLayer.lastIndex) {
            val delta = result[i] - outputLayer[i].output
            error += delta * delta
        }
        error /= outputLayer.lastIndex
        error = sqrt(error)
        error += lambdaW * sqrt(network.sumByDouble {
            it.sumByDouble { it.memory.sumByDouble { it * it } }
        })
        return error
    }

    private fun computeRecentAverageError() = (recentAverageError * recentAverageSmoothingFactor
            + error) / (recentAverageSmoothingFactor + 1.0)


    private fun teachOutputLayer(result: DoubleArray) {
        val outputLayer = network.last()
        for (i in 0 until outputLayer.lastIndex)
            outputLayer[i].learnFromResult(result[i])
    }

    private fun teachLayers() {
        for (i in network.lastIndex - 1 downTo 1) {
            val layer = network[i]
            val nextLayer = network[i + 1]
            layer.forEach { it.learnFromNextLayer(nextLayer) }
        }
    }

    private fun remember() {
        for (i in network.lastIndex downTo 1) {
            val layer = network[i]
            val previousLayer = network[i - 1]
            for (j in 0 until layer.lastIndex) {
                layer[j].remember(previousLayer)
            }
        }
    }

    fun results() = network.last().map { it.output }.dropLast(1).toDoubleArray()

}