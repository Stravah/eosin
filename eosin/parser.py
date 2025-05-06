import logging
import pandas as pd
import pdfplumber
from eosin.utils import combine_text_objects, group_adjacent_text, is_valid_date
import re

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DESIRABLE_HEADERS = [
    "deposit",
    "withdrawal",
    "credit",
    "detail",
    "particular",
    "reference",
    "chq",
    "cheque",
    "narration",
    "remarks",
    "balance",
]


class Parser:
    def __init__(self, statement: str):
        self.statement: str = statement
        self.headers: list[str] = []
        self.data: pd.DataFrame | None = None
        self.pdf_object: pdfplumber.Page | None = None
        self.words_list: list[dict] | None = None
        self.table_date: dict | None = None
        self.date_column_dimensions: tuple[int, int] | None = None
        self.date_rows: list[dict] | None = None
        self.last_row_bottom: float | None = None
        self.all_pages_data: list[pd.DataFrame] = []

    def parse(self) -> pd.DataFrame:
        logger.info("Starting to parse the PDF")
        with pdfplumber.open(self.statement) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"PDF has {page_count} pages")

            table_start_page = -1
            for page_num in range(page_count):
                logger.info(f"Checking page {page_num+1} for table header")
                page = pdf.pages[page_num]
                self.pdf_object = page
                self._get_words()

                if self._check_for_table_headers():
                    table_start_page = page_num
                    logger.info(f"Found table header on page {page_num+1}")
                    break

            if table_start_page == -1:
                logger.error("No table header found in any page of the document")
                return pd.DataFrame()

            self._find_date_header()
            logger.info("Date header found successfully")
            self._find_date_rows()
            logger.info("Date rows identified successfully")

            page_df = self._parse_dates_top_aligned()
            if page_df is not None and not page_df.empty:
                self.all_pages_data.append(page_df)

            if page_count > table_start_page + 1:
                logger.info(
                    f"Processing additional {page_count-(table_start_page+1)} pages"
                )
                for page_num in range(table_start_page + 1, page_count):
                    logger.info(f"Processing page {page_num+1}")
                    page = pdf.pages[page_num]
                    self.pdf_object = page
                    self._get_words()

                    has_headers = self._check_for_table_headers()

                    if has_headers:
                        logger.info(f"Found table headers on page {page_num+1}")
                        self._find_date_header()
                        self._find_date_rows()
                    else:
                        logger.info(
                            f"No table headers found on page {page_num+1}, using dimensions from first table page"
                        )
                        self._find_date_rows_without_headers()

                    # Skip empty pages or pages with no date rows
                    if not self.date_rows or len(self.date_rows) == 0:
                        logger.info(
                            f"No valid date rows found on page {page_num+1}, skipping"
                        )
                        continue

                    # Process the page
                    page_df = self._parse_dates_top_aligned()
                    if page_df is not None and not page_df.empty:
                        self.all_pages_data.append(page_df)

                    # Check if this might be the end of the table
                    if self._detect_end_of_table():
                        # Peek ahead to the next page to see if table continues
                        if page_num + 1 < page_count:
                            # Save current state
                            current_pdf_object = self.pdf_object
                            current_words_list = self.words_list
                            current_date_rows = self.date_rows

                            # Check next page
                            logger.info(
                                f"Detected potential table end on page {page_num+1}, peeking at page {page_num+2}"
                            )
                            next_page = pdf.pages[page_num + 1]
                            self.pdf_object = next_page
                            self._get_words()

                            # If next page has headers, it's likely a continuation
                            next_page_has_headers = self._check_for_table_headers()

                            if next_page_has_headers:
                                logger.info(
                                    f"Found table headers on next page {page_num+2}, continuing"
                                )
                                # Restore state and continue (next iteration will process this page)
                                self.pdf_object = current_pdf_object
                                self.words_list = current_words_list
                                self.date_rows = current_date_rows
                                continue

                            # If next page doesn't have headers, try to find date rows
                            self._find_date_rows_without_headers()

                            if self.date_rows and len(self.date_rows) > 0:
                                logger.info(
                                    f"Found date rows on next page {page_num+2}, continuing"
                                )
                                # Restore state and continue
                                self.pdf_object = current_pdf_object
                                self.words_list = current_words_list
                                self.date_rows = current_date_rows
                                continue

                            # Restore state
                            self.pdf_object = current_pdf_object
                            self.words_list = current_words_list
                            self.date_rows = current_date_rows

                        logger.info(f"End of table confirmed on page {page_num+1}")
                        break

            if self.all_pages_data:
                self.data = pd.concat(self.all_pages_data, ignore_index=True)
                logger.info(
                    f"Successfully parsed {len(self.all_pages_data)} pages with {len(self.data)} total rows"
                )
            else:
                logger.warning("No data was extracted from the PDF")

        return self.data

    def _get_words(self) -> list[dict]:
        words_list: list[dict] = self.pdf_object.extract_words()
        for index, word in enumerate(words_list):
            word["index"] = index

        self.words_list = words_list

    def _find_nearby_headers(
        self, text_date_object: dict, top_padding: int = 15, bottom_padding: int = 10
    ) -> list[dict]:
        words_list: list[dict] = self.words_list

        potential_headers = []

        for _i, word in enumerate(words_list):
            if (
                (word["top"] + top_padding) > text_date_object["top"]
                and (word["bottom"] - bottom_padding) < text_date_object["bottom"]
                and word["x0"] > text_date_object["x0"]
            ):
                potential_headers.append(word)

        logger.debug(
            f"Potential headers found: {[potential_header['text'] for potential_header in potential_headers]}"
        )
        return potential_headers

    def _is_table_date(self, date):

        nearby_headers = self._find_nearby_headers(date)

        for nearby_header in nearby_headers:
            if any(
                desirable_header in nearby_header["text"].lower()
                for desirable_header in DESIRABLE_HEADERS
            ):
                logger.debug(f"Found desirable date: {date}")
                return True
        return False
        return False

    # TODO: properly implement this function
    def _find_date_header_padding(self, date_header):
        logger.info("Finding date header padding")

        header_list = self._find_nearby_headers(date_header)

        horizontal_gaps = []
        for x in range(len(header_list) - 1):
            horizontal_gaps.append(header_list[x + 1]["x0"] - header_list[x]["x1"])

        vertical_gaps = []
        for index, _header in enumerate(header_list):
            vertical_gaps.append(header_list[index]["bottom"])

        logger.debug(f"Horizontal gaps: {horizontal_gaps}")
        logger.debug(f"Vertical gaps: {vertical_gaps}")
        return [30, 30]

    def _find_date_header(self):
        logger.info("Finding date header")

        words_list = self.words_list
        date_header_terms = [
            "date",
            "transaction date",
            "value date",
            "posting date",
            "tran date",
        ]
        table_date = None

        for word in words_list:
            word_text = word["text"].lower()
            if any(term in word_text for term in date_header_terms):
                potential_date = word
                logger.debug(f"Potential date header found: {potential_date}")

                if self._is_table_date(potential_date):
                    table_date = potential_date
                    logger.debug(f"Table date header found: {table_date}")

                    adjacent_headers = self._find_nearby_headers(table_date)
                    headers = group_adjacent_text(adjacent_headers, expected_gap=5)
                    padding = self._find_date_header_padding(table_date)

                    date_column_dimensions = (
                        table_date["x0"] - padding[0],
                        table_date["x1"] + padding[1],
                    )

                    logger.debug(f"Date column dimensions: {date_column_dimensions}")

                    self.table_date = table_date
                    self.date_column_dimensions = date_column_dimensions
                    self.headers = headers
                    return

        if not table_date:
            rows = {}
            for word in words_list:
                row_key = round(word["top"] / 5) * 5
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append(word)

            for row_key, row_words in sorted(rows.items()):
                header_words = []
                for word in row_words:
                    word_text = word["text"].lower()
                    if any(
                        header in word_text
                        for header in DESIRABLE_HEADERS + ["date", "balance"]
                    ):
                        header_words.append(word)

                if len(header_words) >= 3:
                    header_words.sort(key=lambda x: x["x0"])

                    leftmost_header = header_words[0]

                    padding = [30, 30]
                    date_column_dimensions = (
                        leftmost_header["x0"] - padding[0],
                        leftmost_header["x1"] + padding[1],
                    )

                    logger.debug(
                        f"Inferred date column dimensions: {date_column_dimensions}"
                    )

                    headers = group_adjacent_text(header_words, expected_gap=5)

                    self.table_date = leftmost_header
                    self.date_column_dimensions = date_column_dimensions
                    self.headers = headers
                    return

        raise Exception("Date header not found")

    # TODO: Refactor this monstrosity of a method
    def _find_date_rows(self):

        logger.debug("Finding date rows")

        words_list = self.words_list
        date_column_dimensions = self.date_column_dimensions

        logger.debug("Finding potential date rows in the date column")

        potential_dates = []

        table_date_index = self.table_date["index"]

        for i in range(table_date_index + 1, len(words_list)):
            if (
                words_list[i]["x0"] > date_column_dimensions[0]
                and words_list[i]["x1"] < date_column_dimensions[1]
            ):
                potential_dates.append(words_list[i])

        logger.debug(
            f"Potential dates found: {[potential_date['text'] for potential_date in potential_dates]}"
        )

        dates = []
        i = 0

        logger.debug("Validating potential dates")

        while i in range(len(potential_dates)):
            match is_valid_date(potential_dates[i]["text"]):
                case "Valid":
                    dates.append(potential_dates[i])
                case "Incomplete":
                    if i + 1 < len(potential_dates):
                        match is_valid_date(
                            combine_text_objects(
                                [potential_dates[i], potential_dates[i + 1]]
                            )["text"]
                        ):
                            case "Valid":
                                dates.append(
                                    combine_text_objects(
                                        [potential_dates[i], potential_dates[i + 1]]
                                    )
                                )
                                i += 1
                            case "Incomplete":
                                match is_valid_date(
                                    combine_text_objects(
                                        [
                                            potential_dates[i],
                                            potential_dates[i + 1],
                                            potential_dates[i + 2],
                                        ]
                                    )["text"]
                                ):
                                    case "Valid":
                                        dates.append(
                                            combine_text_objects(
                                                [
                                                    potential_dates[i],
                                                    potential_dates[i + 1],
                                                    potential_dates[i + 2],
                                                ]
                                            )
                                        )
                                        i += 2
                                    case "Incomplete":
                                        pass
                                    case "Invalid":
                                        pass
                            case "Invalid":
                                pass
                case "Invalid":
                    pass
            i += 1

        logger.debug(f"Dates found: {[date['text'] for date in dates]}")
        self.date_rows = dates

    def _categorize_text_into_headers(self, text_objects):
        """Categorise each text object into one of the detected column headers.

        This is the original, overlap-based approach that worked well for most
        columns (Date, Value Date, Particulars, etc.).  We are restoring it to
        avoid the recently-observed mix-ups between date and description.  The
        small-amount / balance merge problem is now handled earlier (during row
        token extraction) by physically splitting a merged numeric string into
        two distinct text objects, so this routine can stay simple.
        """
        logger.debug("Categorizing text into headers (overlap heuristic)")

        headers = self.headers

        categorized_text = {header["text"]: "" for header in headers}

        for text in text_objects:
            placed = False
            for header in headers:
                # Check horizontal overlap with header bbox (+/- widths)
                if (
                    max(header["x1"], text["x1"]) - min(header["x0"], text["x0"])
                    < header["width"] + text["width"]
                ):
                    categorized_text[header["text"]] += text["text"]
                    placed = True
                    break
            if not placed:
                # If no overlap match, put the text in the first header that is to the right
                for header in headers:
                    if header["x0"] > text["x0"]:
                        categorized_text[header["text"]] += text["text"]
                        break
        logger.debug(f"Categorized text: {categorized_text}")
        return categorized_text

    @staticmethod
    def _split_merged_amount(text: str) -> tuple[str, str] | None:
        """Detect patterns like '1.0065527.50' and split them into
        ('1.00', '65527.50').  Returns None if the text is not a merged number."""
        # Must have exactly two dots
        if text.count(".") != 2:
            return None
        # Basic split using first dot position
        first_dot = text.find(".")
        if first_dot == -1 or first_dot + 3 >= len(text):
            return None
        first_number = text[: first_dot + 3]  # up to 2 decimal digits
        remainder = text[first_dot + 3 :]
        # remainder should contain another '.' separating integral and decimals
        second_dot = remainder.find(".")
        if second_dot == -1 or second_dot + 3 != len(remainder):
            return None
        second_number = remainder[: second_dot] + "." + remainder[second_dot + 1 :]
        # Sanity: both parts should be numeric
        num_pattern = re.compile(r"^[0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}$|^[0-9]+\.[0-9]{2}$")
        if num_pattern.match(first_number) and num_pattern.match(second_number):
            return first_number, second_number
        return None

    def _find_date_rows_without_headers(self):
        """Find date rows on pages without headers using dimensions from first page"""
        logger.info("Finding date rows without headers")

        if not self.date_column_dimensions:
            logger.error("No date column dimensions available from first page")
            return

        words_list = self.words_list
        date_column_dimensions = self.date_column_dimensions

        potential_dates = []

        for i, word in enumerate(words_list):
            word["index"] = i
            if (
                word["x0"] > date_column_dimensions[0]
                and word["x1"] < date_column_dimensions[1]
            ):
                potential_dates.append(word)

        potential_dates.sort(key=lambda x: x["top"])

        logger.debug(
            f"Potential dates found: {[potential_date['text'] for potential_date in potential_dates]}"
        )

        dates = []
        i = 0

        while i < len(potential_dates):
            match is_valid_date(potential_dates[i]["text"]):
                case "Valid":
                    dates.append(potential_dates[i])
                case "Incomplete":
                    if i + 1 < len(potential_dates):
                        match is_valid_date(
                            combine_text_objects(
                                [potential_dates[i], potential_dates[i + 1]]
                            )["text"]
                        ):
                            case "Valid":
                                dates.append(
                                    combine_text_objects(
                                        [potential_dates[i], potential_dates[i + 1]]
                                    )
                                )
                                i += 1
                            case "Incomplete":
                                if i + 2 < len(potential_dates):
                                    match is_valid_date(
                                        combine_text_objects(
                                            [
                                                potential_dates[i],
                                                potential_dates[i + 1],
                                                potential_dates[i + 2],
                                            ]
                                        )["text"]
                                    ):
                                        case "Valid":
                                            dates.append(
                                                combine_text_objects(
                                                    [
                                                        potential_dates[i],
                                                        potential_dates[i + 1],
                                                        potential_dates[i + 2],
                                                    ]
                                                )
                                            )
                                            i += 2
                                        case "Incomplete":
                                            pass
                                        case "Invalid":
                                            pass
                            case "Invalid":
                                pass
                case "Invalid":
                    pass
            i += 1

        dates.sort(key=lambda x: x["top"])
        logger.debug(f"Dates found without headers: {[date['text'] for date in dates]}")
        self.date_rows = dates

    def _parse_dates_top_aligned(self):

        logger.debug("Parsing dates with multi-alignment support")

        dates = self.date_rows
        words_list = self.words_list

        if not dates or len(dates) <= 1:
            logger.warning("Insufficient date rows found for parsing")
            return None

        if self.table_date:
            table_date_index = self.table_date["index"]
        else:
            table_date_index = 0

        row_boundaries = []
        for i in range(len(dates)):
            if i < len(dates) - 1:
                current_bottom = dates[i]["bottom"]
                next_top = dates[i + 1]["top"]
                row_bottom = (current_bottom + next_top) / 2
            else:
                page_height = (
                    max([word["bottom"] for word in words_list]) if words_list else 800
                )

                if i > 0:
                    avg_row_height = (dates[i]["bottom"] - dates[i]["top"]) * 1.5
                    row_bottom = min(dates[i]["bottom"] + avg_row_height, page_height)
                else:
                    row_bottom = min(dates[i]["bottom"] + 40, page_height)

            row_boundaries.append((dates[i]["top"] - 2, row_bottom))

        logger.debug(f"Row boundaries: {row_boundaries}")

        table = {}

        first_page_columns = []
        column_positions = []
        if self.all_pages_data and not self.all_pages_data[0].empty:
            first_page_columns = list(self.all_pages_data[0].columns)

            if self.headers:
                column_positions = [(h["x0"], h["x1"]) for h in self.headers]

        for i, date in enumerate(dates):
            date_text = date["text"]
            row_top, row_bottom = row_boundaries[i]

            row_text_elements = []

            for j in range(len(words_list)):
                word = words_list[j]
                if word["index"] == date["index"]:
                    continue

                if word["x0"] <= date["x0"]:
                    continue

                # Strategy 1: Text is fully contained within row boundaries (top alignment)
                fully_contained = (
                    word["top"] >= row_top and word["bottom"] <= row_bottom
                )

                # Strategy 2: Text center is within row boundaries (center alignment)
                word_center = (word["top"] + word["bottom"]) / 2
                center_aligned = word_center >= row_top and word_center <= row_bottom

                # Strategy 3: Text significantly overlaps with row (any alignment)
                overlap_height = min(row_bottom, word["bottom"]) - max(
                    row_top, word["top"]
                )
                if word["bottom"] > word["top"]:  # Avoid division by zero
                    overlap_ratio = overlap_height / (word["bottom"] - word["top"])
                else:
                    overlap_ratio = 0
                significant_overlap = overlap_height > 0 and overlap_ratio > 0.3

                # Include the text if any alignment strategy matches
                if fully_contained or center_aligned or significant_overlap:
                    row_text_elements.append(word)

            # Continue with existing grouping logic for compatibility
            potential_headers = [date]
            grouped_row_text = group_adjacent_text(
                potential_headers + row_text_elements
            )

            if grouped_row_text and grouped_row_text[0]["x0"] <= date["x1"]:
                grouped_row_text = grouped_row_text[1:]

            if self.headers:
                categorized_text = self._categorize_text_into_headers(grouped_row_text)

                # ---------------------------------------------------------
                # Fix potential merged numeric strings in this single row
                # ---------------------------------------------------------
                categorized_text = self._fix_row_merged_numeric(categorized_text)

                # Handle Date column
                if len(table) == 0 and "Date" not in categorized_text:
                    table["Date"] = []

                if "Date" in table and "Date" not in categorized_text:
                    if len(table["Date"]) < len(list(table.values())[0]) + 1:
                        table["Date"].append(date_text)

                # Add the rest of the columns
                for ctext in categorized_text:
                    if ctext not in table:
                        table[ctext] = [categorized_text[ctext]]
                    else:
                        table[ctext].append(categorized_text[ctext])
            else:
                row_elements = sorted(grouped_row_text, key=lambda x: x["x0"])
                row_data = {}

                if first_page_columns:
                    date_column = first_page_columns[0]
                    row_data[date_column] = date_text

                    if column_positions:
                        for elem in row_elements:
                            for idx, (col_x0, col_x1) in enumerate(column_positions):
                                if idx > 0 and (
                                    (
                                        elem["x0"] >= col_x0 - 20
                                        and elem["x0"] <= col_x1 + 20
                                    )
                                    or (
                                        elem["x1"] >= col_x0 - 20
                                        and elem["x1"] <= col_x1 + 20
                                    )
                                    or (elem["x0"] <= col_x0 and elem["x1"] >= col_x1)
                                ):
                                    if idx < len(first_page_columns):
                                        col_name = first_page_columns[idx]
                                        if col_name in row_data:
                                            row_data[col_name] += " " + elem["text"]
                                        else:
                                            row_data[col_name] = elem["text"]
                                    break
                    else:
                        # Use horizontal position to assign columns
                        start_idx = 0 if date_column not in row_data else 1
                        for idx, elem in enumerate(row_elements):
                            col_idx = idx + start_idx
                            if col_idx < len(first_page_columns):
                                col_name = first_page_columns[col_idx]
                                if col_name != date_column:
                                    if col_name in row_data:
                                        row_data[col_name] += " " + elem["text"]
                                    else:
                                        row_data[col_name] = elem["text"]

                    # Add row data to table
                    for col in first_page_columns:
                        if col not in table:
                            table[col] = [row_data.get(col, "")]
                        else:
                            table[col].append(row_data.get(col, ""))

        # Store position of last row for next page processing
        if dates and len(dates) > 0:
            self.last_row_bottom = max(date["bottom"] for date in dates)

        # Ensure all columns have the same length
        if table:
            max_length = max(len(values) for values in table.values())
            for col in table:
                if len(table[col]) < max_length:
                    table[col].extend([""] * (max_length - len(table[col])))

            df = pd.DataFrame(table)
            return df
        else:
            return None

    def _check_for_table_headers(self) -> bool:
        logger.info("Checking for table headers on current page")

        header_terms = [
            "date",
            "transaction date",
            "value date",
            "description",
            "narration",
            "particulars",
            "details",
            "chq/ref no",
            "cheque",
            "withdrawal",
            "debit",
            "credit",
            "deposit",
            "amount",
            "balance",
        ]

        words_list = self.words_list

        rows = {}
        for word in words_list:
            row_key = round(word["top"] / 5) * 5
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(word)

        for row_key, row_words in rows.items():
            header_matches = 0
            for word in row_words:
                word_text = word["text"].lower()
                if any(term in word_text for term in header_terms):
                    header_matches += 1
                    # Check specifically for date header with typical statement columns
                    if "date" in word_text:
                        potential_date = word
                        if self._is_table_date(potential_date):
                            logger.debug(f"Found table date header: {word['text']}")
                            return True

            if header_matches >= 3:
                logger.debug(f"Found header row with {header_matches} header terms")
                return True

        return False

    def _detect_end_of_table(self) -> bool:

        if not self.date_rows or len(self.date_rows) == 0:
            # No date rows found on this page
            return True

        if len(self.date_rows) < 3:
            # Get page dimensions to check if dates are at the bottom of the page
            page_height = (
                max([word["bottom"] for word in self.words_list])
                if self.words_list
                else 800
            )
            last_date_bottom = max(date["bottom"] for date in self.date_rows)

            # If the last date is near the bottom of the page, likely continues on next page
            if last_date_bottom > (page_height * 0.8):  # Bottom 20% of page
                logger.debug(
                    f"Few dates ({len(self.date_rows)}) found near page bottom, likely continues on next page"
                )
                return False

        # Check for ending keywords typically found at the end of bank statements
        ending_keywords = [
            "closing balance",
            "total",
            "balance c/f",
            "balance b/f",
            "grand total",
            "end of statement",
            "closing value",
            "statement end",
        ]

        # More definitive ending phrases that strongly indicate end of statement
        strong_ending_phrases = [
            "end of statement",
            "statement end",
            "this is the end of the statement",
            "for customer service",
            "for any assistance",
            "visit our website",
            "thank you for banking with us",
        ]

        words_list = self.words_list

        # Combine adjacent words to form phrases for better matching
        text_blocks = []
        for i in range(len(words_list) - 1):
            if (
                abs(words_list[i]["bottom"] - words_list[i + 1]["bottom"]) < 5
            ):  # Same line
                text_blocks.append(
                    words_list[i]["text"] + " " + words_list[i + 1]["text"]
                )

        # Find the bottom of the last date row
        if self.date_rows:
            last_date_bottom = max(date["bottom"] for date in self.date_rows)

            # Look for ending keywords below the last date row
            found_ending_keyword = False
            for word in words_list:
                if word["top"] > last_date_bottom:
                    word_text = word["text"].lower()

                    # Check for strong ending phrases first (more definitive)
                    if any(phrase in word_text for phrase in strong_ending_phrases):
                        logger.debug(f"Found strong ending phrase: {word['text']}")
                        return True

                    # Check for regular ending keywords
                    if any(keyword in word_text for keyword in ending_keywords):
                        found_ending_keyword = True
                        logger.debug(f"Found ending keyword: {word['text']}")

            # For regular ending keywords, require additional validation
            if found_ending_keyword:
                # Check if there's a significant amount of content after the keyword
                # If there is, it's likely not just a footer but actual end of table
                content_after_last_date = [
                    w for w in words_list if w["top"] > last_date_bottom
                ]
                if len(content_after_last_date) > 10:
                    return True

            # Check for large gap between dates (might indicate end of data)
            if self.last_row_bottom is not None:
                # If the first date on this page is significantly below where we'd expect,
                # it could indicate end of meaningful table data
                first_date_top = min(date["top"] for date in self.date_rows)
                if (
                    first_date_top > 100
                ):  # Arbitrary threshold - adjust based on your PDFs
                    logger.debug(
                        f"Large gap detected before first date: {first_date_top}"
                    )
                    # Look at content in this gap - if it's substantial, likely end of table
                    content_in_gap = [
                        w
                        for w in words_list
                        if w["top"] < first_date_top and w["bottom"] > 50
                    ]
                    if (
                        len(content_in_gap) > 10
                    ):  # If there's substantial content before dates
                        return True

        # By default, assume the table continues unless we have strong evidence otherwise
        return False

    # -------------------------------------------------------------
    # Helpers for numeric merge at row level (small debit amounts)
    # -------------------------------------------------------------
    def _fix_row_merged_numeric(self, row_dict: dict[str, str]) -> dict[str, str]:
        """Detect and separate merged debit/credit numbers from balance within
        a single row dictionary produced by _categorize_text_into_headers.

        If the Balance column contains a string with *two* decimal points and
        one of the primary amount columns (Withdrawals / Deposits / Debit /
        Credit) is empty, attempt to split the merged string and populate the
        empty amount column accordingly.
        """

        # Identify column names
        balance_col = None
        amount_cols: list[str] = []
        for col in row_dict:
            col_l = col.lower()
            if any(k in col_l for k in ["balance", "closing"]):
                balance_col = col
            elif any(k in col_l for k in ["withdrawal", "withdrawals", "debit", "deposit", "credit"]):
                amount_cols.append(col)

        if not balance_col or not amount_cols:
            return row_dict  # Nothing to do

        bal_val = row_dict.get(balance_col, "")
        if not bal_val or bal_val.count(".") != 2:
            return row_dict  # No merge pattern

        split_res = self._split_merged_amount(bal_val)
        if not split_res:
            return row_dict

        amount_part, balance_part = split_res

        # Choose first empty amount column to fill
        target_amount_col = None
        for ac in amount_cols:
            if not row_dict.get(ac):
                target_amount_col = ac
                break

        if not target_amount_col:
            return row_dict  # All amount cols already populated

        row_dict[target_amount_col] = amount_part
        row_dict[balance_col] = balance_part
        return row_dict
