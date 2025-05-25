package org.example;

import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

class ImageComparisonPanel extends JPanel {
    private JButton loadImagesBtn;
    private JButton compareBtn;
    private JLabel resultLabel;
    private List<Mat> loadedImages = new ArrayList<>();

    public ImageComparisonPanel() {
        setLayout(new BorderLayout());

        JPanel buttonPanel = new JPanel();
        loadImagesBtn = new JButton("Load Images (3-10)");
        compareBtn = new JButton("Compare");

        buttonPanel.add(loadImagesBtn);
        buttonPanel.add(compareBtn);

        add(buttonPanel, BorderLayout.NORTH);

        resultLabel = new JLabel("Results will be shown here");
        add(new JScrollPane(resultLabel), BorderLayout.CENTER);

        loadImagesBtn.addActionListener(e -> loadImages());
        compareBtn.addActionListener(e -> compareImages());
    }

    private void loadImages() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setMultiSelectionEnabled(true);

        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            loadedImages.clear();
            File[] files = fileChooser.getSelectedFiles();

            if (files.length < 3 || files.length > 10) {
                JOptionPane.showMessageDialog(this, "Please select between 3 and 10 images.");
                return;
            }

            for (File file : files) {
                Mat img = Imgcodecs.imread(file.getAbsolutePath());
                if (!img.empty()) {
                    loadedImages.add(img);
                }
            }

            resultLabel.setText("Loaded " + loadedImages.size() + " images.");
        }
    }

    private void compareImages() {
        if (loadedImages.size() < 3) {
            JOptionPane.showMessageDialog(this, "Please load at least 3 images first.");
            return;
        }

        // Используем SIFT для извлечения признаков
        SIFT sift = SIFT.create();
        List<MatOfKeyPoint> keypoints = new ArrayList<>();
        List<Mat> descriptors = new ArrayList<>();

        for (Mat img : loadedImages) {
            Mat gray = new Mat();
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

            MatOfKeyPoint kp = new MatOfKeyPoint();
            Mat desc = new Mat();
            sift.detectAndCompute(gray, new Mat(), kp, desc);

            keypoints.add(kp);
            descriptors.add(desc);
        }

        // Сравниваем все изображения попарно
        double maxSimilarity = -1;
        int bestPair1 = -1, bestPair2 = -1;

        BFMatcher matcher = BFMatcher.create(BFMatcher.BRUTEFORCE, true);

        for (int i = 0; i < descriptors.size(); i++) {
            for (int j = i + 1; j < descriptors.size(); j++) {
                MatOfDMatch matches = new MatOfDMatch();
                matcher.match(descriptors.get(i), descriptors.get(j), matches);

                // Фильтрация хороших совпадений
                List<DMatch> matchesList = matches.toList();
                matchesList.sort((a, b) -> Float.compare(a.distance, b.distance));

                double similarity = 0;
                int goodMatches = 0;
                for (int k = 0; k < Math.min(50, matchesList.size()); k++) {
                    similarity += matchesList.get(k).distance;
                    goodMatches++;
                }

                similarity = goodMatches > 0 ? similarity / goodMatches : Double.MAX_VALUE;

                if (maxSimilarity == -1 || similarity < maxSimilarity) {
                    maxSimilarity = similarity;
                    bestPair1 = i;
                    bestPair2 = j;
                }
            }
        }

        // Показываем результаты
        if (bestPair1 != -1 && bestPair2 != -1) {
            resultLabel.setText(String.format("Most similar images: %d and %d (similarity: %.2f)",
                    bestPair1 + 1, bestPair2 + 1, maxSimilarity));

            // Показываем изображения
            Mat img1 = loadedImages.get(bestPair1);
            Mat img2 = loadedImages.get(bestPair2);

            // Создаем мозаику из двух изображений
            Mat combined = new Mat(Math.max(img1.rows(), img2.rows()), img1.cols() + img2.cols(), img1.type());

            Mat left = combined.submat(0, img1.rows(), 0, img1.cols());
            img1.copyTo(left);

            Mat right = combined.submat(0, img2.rows(), img1.cols(), img1.cols() + img2.cols());
            img2.copyTo(right);

            BufferedImage resultImage = matToBufferedImage(combined);
            resultLabel.setIcon(new ImageIcon(resultImage));
            resultLabel.setText("");
        }
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        // Определяем тип BufferedImage в зависимости от количества каналов в Mat
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        // Получаем размеры изображения
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];

        // Копируем данные из Mat в byte array
        mat.get(0, 0, buffer);

        // Создаем BufferedImage
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);

        // Получаем доступ к пикселям BufferedImage
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        // Копируем данные из buffer в targetPixels
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);

        return image;
    }
}