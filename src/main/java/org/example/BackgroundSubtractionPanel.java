package org.example;

import org.opencv.core.Mat;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

class BackgroundSubtractionPanel extends JPanel {
    private JButton loadVideoBtn;
    private JButton startBtn;
    private JLabel videoLabel;
    private VideoCapture videoCapture;
    private boolean isRunning;
    private BackgroundSubtractorMOG2 subtractor;

    public BackgroundSubtractionPanel() {
        setLayout(new BorderLayout());

        JPanel buttonPanel = new JPanel();
        loadVideoBtn = new JButton("Load Video");
        startBtn = new JButton("Start/Stop");

        buttonPanel.add(loadVideoBtn);
        buttonPanel.add(startBtn);

        add(buttonPanel, BorderLayout.NORTH);

        videoLabel = new JLabel();
        add(new JScrollPane(videoLabel), BorderLayout.CENTER);

        subtractor = Video.createBackgroundSubtractorMOG2();

        loadVideoBtn.addActionListener(e -> loadVideo());
        startBtn.addActionListener(e -> toggleProcessing());
    }

    private void loadVideo() {
        JFileChooser fileChooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "Video Files", "mp4", "avi", "mov", "mkv");
        fileChooser.setFileFilter(filter);

        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                // Получаем выбранный файл
                File videoFile = fileChooser.getSelectedFile();

                // Конвертируем путь в формат, понятный OpenCV
                String videoPath = videoFile.getAbsolutePath();

                // Создаем временную копию с ASCII-именем, если путь содержит Unicode
                if (!isAscii(videoPath)) {
                    File tempFile = createTempVideoCopy(videoFile);
                    videoPath = tempFile.getAbsolutePath();
                }

                // Инициализируем VideoCapture
                videoCapture = new VideoCapture();
                if (!videoCapture.open(videoPath)) {
                    throw new Exception("Failed to open video file");
                }

                JOptionPane.showMessageDialog(this,
                        "Video loaded successfully: " + videoFile.getName(),
                        "Success", JOptionPane.INFORMATION_MESSAGE);

            } catch (Exception e) {
                JOptionPane.showMessageDialog(this,
                        "Error loading video:\n" + e.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    // Проверка на ASCII-символы
    private boolean isAscii(String path) {
        return path.matches("\\A\\p{ASCII}*\\z");
    }

    // Создание временной копии видеофайла
    private File createTempVideoCopy(File originalFile) throws IOException {
        String tempDir = System.getProperty("java.io.tmpdir");
        String tempFileName = "video_" + System.currentTimeMillis() +
                getFileExtension(originalFile.getName());

        File tempFile = new File(tempDir, tempFileName);

        Files.copy(originalFile.toPath(), tempFile.toPath(),
                StandardCopyOption.REPLACE_EXISTING);

        tempFile.deleteOnExit();
        return tempFile;
    }

    // Получение расширения файла
    private String getFileExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        return (dotIndex == -1) ? "" : fileName.substring(dotIndex);
    }

    private void toggleProcessing() {
        if (videoCapture == null || !videoCapture.isOpened()) {
            JOptionPane.showMessageDialog(this, "Please load a video first.");
            return;
        }

        if (isRunning) {
            stopProcessing();
        } else {
            startProcessing();
        }
    }

    private void startProcessing() {
        isRunning = true;
        new Thread(() -> {
            Mat frame = new Mat();
            Mat fgMask = new Mat();

            while (isRunning && videoCapture.read(frame)) {
                subtractor.apply(frame, fgMask);

                Mat result = new Mat();
                frame.copyTo(result, fgMask);

                BufferedImage image = matToBufferedImage(result);
                SwingUtilities.invokeLater(() -> {
                    videoLabel.setIcon(new ImageIcon(image));
                });

                try {
                    Thread.sleep(30); // ~30 FPS
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            if (isRunning) {
                // Видео закончилось
                isRunning = false;
                videoCapture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
            }
        }).start();
    }

    private void stopProcessing() {
        isRunning = false;
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