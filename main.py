import pattern
import generator

checkers = pattern.Checker(200, 10)
checkers.draw()
checkers.show()


circle = pattern.Circle(2048, 400, (512, 256))
circle.draw()
circle.show()

image = generator.ImageGenerator('./exercise_data/', './Labels.json', 5, [32, 32, 3], rotation=False, mirroring=True, shuffle=False)
image.next()
image.show()
