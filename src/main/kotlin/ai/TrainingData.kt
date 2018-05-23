package ai

import java.io.File

class TrainingData(private val file: File) {
    val topology: Array<Int>
        get() {
            file.useLines {
                return it.first().split(" ").map { it.toInt() }.toTypedArray()
            }

        }

    val inputs get() = collectLinesByPrefix("in: ")

    val outputs get() = collectLinesByPrefix("out: ")

    private fun collectLinesByPrefix(prefix: String) =
            file.readLines().asSequence().filter { it.startsWith(prefix) }
                    .map {
                        it.split(" ").drop(1).map { it.toDouble() }.toTypedArray()
                    }.toList()
}