import pattern
import generator


checkers = pattern.Checker(250, 25)
checkers.draw()
checkers.show()


circle = pattern.Circle(1024, 200, (512, 256))
circle.draw()
circle.show()

spectrum = pattern.Spectrum(256)
spectrum.draw()
spectrum.show()

image = generator.ImageGenerator('./exercise_data/', './Labels.json', 12, [50, 50, 3], rotation=False, mirroring=True, shuffle=False)
image.next()[0]
image.show()
