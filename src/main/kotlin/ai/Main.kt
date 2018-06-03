package ai

import com.sun.media.sound.InvalidFormatException
import java.io.File
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*
import kotlin.math.roundToInt

val trainingFile = "testData.txt"

fun generateXorTrainingData(fileName: String, sampleCount: Int): File {
    val lines = mutableListOf<String>()
    lines += "2 4 1"
    for (i in sampleCount downTo 0) {
        val n1 = (2.0 * Math.random()).toInt()
        val n2 = (2.0 * Math.random()).toInt()
        val o = n1 xor n2
        lines += "in: $n1 $n2"
        lines += "out: $o"
    }
    val path = Files.write(Paths.get(fileName), lines, Charset.defaultCharset())
    return File(path.toUri())
}

fun testNetwork(trainingData: TrainingData, brain: Brain): Int {
    val inputs = trainingData.inputs
    val outputs = trainingData.outputs

    if (inputs.size != outputs.size) throw InvalidFormatException("Corrupted training data. Number " +
            "of inputs does not match number of outputs.")

    var lastWrongPass = -1

    for (i in 0 until inputs.size) {
        println("Pass: $i")

        println("Input: ${Arrays.toString(inputs[i])}")
        brain.think(inputs[i])

        val results = brain.results()
        println("NN output: ${Arrays.toString(results)}")

        println("True output: ${Arrays.toString(outputs[i])}")
        brain.learn(outputs[i])

        if (!(results.map { it.roundToInt() }.toIntArray()
                        contentEquals
                        outputs[i].map { it.roundToInt() }.toIntArray()))
            lastWrongPass = i

        println("Recent average error: ${brain.recentAverageError}\n")

    }

    println("Last missed pass: $lastWrongPass")
    return lastWrongPass
}


fun main(args: Array<String>) {
    // Generate file with 2000 xor samples
//    generateXorTrainingData(trainingFile, 2000)
    val trainingData = TrainingData(File(trainingFile))
    val topology = trainingData.topology

    val activationFunction = SimpleActivationFunction.TANH // TANH works best by far
    val brain = Brain(activationFunction, topology)

    testNetwork(trainingData, brain)

}