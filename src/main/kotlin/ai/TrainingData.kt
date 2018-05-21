package ai

import java.io.File
import java.util.stream.Collectors

class TrainingData(private val file: File) {
    val topology: IntArray
        get() {
            file.useLines {
                return it.first().split(" ").map { it.toInt() }.toIntArray()
            }

        }

    val inputs get() = collectLinesByPrefix("in: ")

    val outputs get() = collectLinesByPrefix("out: ")

    private fun collectLinesByPrefix(prefix: String) =
            file.readLines().stream().filter { it.startsWith(prefix) }
                    .map {
                        it.split(" ").drop(1).map { it.toDouble() }.toDoubleArray()
                    }.collect(Collectors.toList())
}