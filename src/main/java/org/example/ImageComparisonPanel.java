package org.example;

import org.opencv.core.*;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ImageComparisonPanel extends JPanel {
        private JLabel resultLabel;
        private List<Mat> loadedImages = new ArrayList<>();
        private JScrollPane scrollPane;

        public ImageComparisonPanel() {
            setLayout(new BorderLayout());

            JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 10));
            JButton loadImagesBtn = new JButton("Load Images (3-10)");
            JButton compareBtn = new JButton("Compare");

            buttonPanel.add(loadImagesBtn);
            buttonPanel.add(compareBtn);
            add(buttonPanel, BorderLayout.NORTH);

            resultLabel = new JLabel();
            resultLabel.setHorizontalAlignment(JLabel.CENTER);
            scrollPane = new JScrollPane(resultLabel);
            scrollPane.setPreferredSize(new Dimension(800, 600));
            add(scrollPane, BorderLayout.CENTER);

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
                    JOptionPane.showMessageDialog(this,
                            "Please select between 3 and 10 images.",
                            "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                for (File file : files) {
                    Mat img = Imgcodecs.imread(file.getAbsolutePath());
                    if (!img.empty()) {
                        loadedImages.add(img);
                    }
                }

                resultLabel.setText("Loaded " + loadedImages.size() + " images.");
                resultLabel.setIcon(null);
            }
        }

        private void compareImages() {
            if (loadedImages.size() < 3) {
                JOptionPane.showMessageDialog(this,
                        "Please load at least 3 images first.",
                        "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            SIFT sift = SIFT.create();
            List<MatOfKeyPoint> keypointsList = new ArrayList<>();
            List<Mat> descriptorsList = new ArrayList<>();

            for (Mat img : loadedImages) {
                Mat gray = new Mat();
                Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

                MatOfKeyPoint keypoints = new MatOfKeyPoint();
                Mat descriptors = new Mat();
                sift.detectAndCompute(gray, new Mat(), keypoints, descriptors);

                keypointsList.add(keypoints);
                descriptorsList.add(descriptors);
            }

            BFMatcher matcher = BFMatcher.create(BFMatcher.BRUTEFORCE);
            double minDistance = Double.MAX_VALUE;
            int bestIdx1 = -1, bestIdx2 = -1;

            for (int i = 0; i < descriptorsList.size(); i++) {
                for (int j = i + 1; j < descriptorsList.size(); j++) {
                    MatOfDMatch matches = new MatOfDMatch();
                    matcher.match(descriptorsList.get(i), descriptorsList.get(j), matches);

                    double totalDistance = 0;
                    List<DMatch> matchesList = matches.toList();
                    for (DMatch match : matchesList) {
                        totalDistance += match.distance;
                    }
                    double avgDistance = totalDistance / matchesList.size();

                    if (avgDistance < minDistance) {
                        minDistance = avgDistance;
                        bestIdx1 = i;
                        bestIdx2 = j;
                    }
                }
            }

            if (bestIdx1 != -1 && bestIdx2 != -1) {
                displayComparisonResult(loadedImages.get(bestIdx1), loadedImages.get(bestIdx2),
                        keypointsList.get(bestIdx1), keypointsList.get(bestIdx2));
            }
        }

    private void displayComparisonResult(Mat img1, Mat img2,
                                         MatOfKeyPoint kp1, MatOfKeyPoint kp2) {
        try {
            Mat rgbImg1 = new Mat();
            Mat rgbImg2 = new Mat();
            Imgproc.cvtColor(img1, rgbImg1, Imgproc.COLOR_BGR2RGB);
            Imgproc.cvtColor(img2, rgbImg2, Imgproc.COLOR_BGR2RGB);

            SIFT sift = SIFT.create();
            Mat descriptors1 = new Mat();
            Mat descriptors2 = new Mat();
            sift.compute(rgbImg1, kp1, descriptors1);
            sift.compute(rgbImg2, kp2, descriptors2);

            BFMatcher matcher = BFMatcher.create();
            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(descriptors1, descriptors2, matches);

            List<DMatch> matchesList = matches.toList();
            matchesList.sort(Comparator.comparingDouble(d -> d.distance));
            List<DMatch> goodMatches = matchesList.subList(0, Math.min(50, matchesList.size()));

            Mat outputImg = new Mat();
            Features2d.drawMatches(
                    rgbImg1, kp1, rgbImg2, kp2,
                    new MatOfDMatch(goodMatches.toArray(new DMatch[0])),
                    outputImg,
                    new Scalar(0, 255, 0),
                    new Scalar(255, 0, 0),
                    new MatOfByte(),
                    Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            );

            Mat resizedImg = new Mat();
            double scale = calculateOptimalScale(outputImg.width(), outputImg.height(),
                    scrollPane.getWidth(), scrollPane.getHeight());
            Imgproc.resize(outputImg, resizedImg, new Size(), scale, scale, Imgproc.INTER_AREA);

            resultLabel.setIcon(new ImageIcon(matToBufferedImage(resizedImg)));
            resultLabel.setText(null);

            rgbImg1.release();
            rgbImg2.release();
            descriptors1.release();
            descriptors2.release();
            outputImg.release();
            resizedImg.release();
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this,
                    "Error drawing matches: " + e.getMessage(),
                    "Error", JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }

        private double calculateOptimalScale(int imgWidth, int imgHeight,
                                             int panelWidth, int panelHeight) {
            double widthScale = (panelWidth * 0.9) / imgWidth;
            double heightScale = (panelHeight * 0.9) / imgHeight;
            return Math.min(widthScale, heightScale);
        }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = mat.channels() > 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);

        byte[] data = new byte[mat.cols() * mat.rows() * mat.channels()];
        mat.get(0, 0, data);

        if (mat.channels() == 3) {
            for (int i = 0; i < data.length; i += 3) {
                byte temp = data[i];
                data[i] = data[i + 2];
                data[i + 2] = temp;
            }
        }

        System.arraycopy(data, 0,
                ((DataBufferByte) image.getRaster().getDataBuffer()).getData(),
                0, data.length);

        return image;
    }
    }