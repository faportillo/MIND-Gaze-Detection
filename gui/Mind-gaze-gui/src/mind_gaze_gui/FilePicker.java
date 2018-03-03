package mind_gaze_gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

public class FilePicker implements ActionListener {
	String path;
	File selectedFile;

	@Override
	public void actionPerformed(ActionEvent e) {
		JFileChooser file = new JFileChooser();
		FileNameExtensionFilter filter = new FileNameExtensionFilter("MOV, MPG & MP4 Videos", "mov", "mpg", "mp4");
		file.setFileFilter(filter);
		int result = file.showOpenDialog(null);
		if (result == JFileChooser.APPROVE_OPTION) {
			System.out.println("You chose to open this file: " + file.getSelectedFile().getName());
			selectedFile = file.getSelectedFile();
			path = selectedFile.getAbsolutePath();
		}
	}

	public String getFilePath() {
		return path;
	}

	public File getFile() {
		return selectedFile;
	}

}
