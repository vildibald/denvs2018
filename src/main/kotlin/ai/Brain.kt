package ai

import java.lang.Math.pow
import kotlin.math.sqrt


typealias Layer = List<Neuron>
typealias Network = List<Layer>

private const val SMOOTHING_FACTOR = 100.0

class Brain(activationFunction: ActivationFunction, topology: Array<Int>) {
    private var error = 0.0
    var recentAverageError = 0.0
    private val network: Network

    private val lambdaW = 0.001


    init {
        val layerCount = topology.size
        network = Array(size = layerCount, init = {
            // it znamena i-tu vrstvu
            val neuronCount = topology[it] + 1
            val outputCount = if (it == layerCount - 1) 0 else topology[it + 1]
            Array(size = neuronCount, init = {
                Neuron(it, activationFunction, outputCount)
            }).toList()
        }).toList()
    }

    fun think(inputs: Array<Double>) {
        network[0].zip(inputs).forEach { (neuron, input) -> neuron.output = input }
        network.zipWithNext().forEach { (previousLayer, currentLayer) ->
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

    private fun computeError(correctResult: Array<Double>): Double {
        val outputLayer = network.last()
        var error = outputLayer.dropLast(1).zip(correctResult).sumByDouble { (neuron, result) ->
            pow(result - neuron.output, 2.0)
        }
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
        network.last().dropLast(1).zip(result).forEach { (neuron, correctResult) ->
            neuron.learnFromResult(correctResult)
        }
    }

    private fun teachLayers() {
        network.drop(1).zipWithNext().asReversed().forEach { (layer, nextLayer) ->
            layer.forEach { it.learnFromNextLayer(nextLayer) }
        }
    }

    private fun remember() {
        network.zipWithNext().asReversed().forEach { (previousLayer, layer)->
            layer.dropLast(1).forEach { it.remember(previousLayer) }
        }
    }

    fun results() = network.last().map { it.output }.dropLast(1).toTypedArray()

}